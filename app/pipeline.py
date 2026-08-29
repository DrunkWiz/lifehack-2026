"""
Core logic, in the order the app calls it:

1. cluster_products         -> group similar products, batched in chunks of 20
2. determine_expected_attrs -> per cluster, what attributes SHOULD exist for this category
3. attribute_completeness   -> per cluster, % of expected attributes actually present
4. suggest_personas         -> 2-3 candidate personas per cluster, each rated on how well
                                the PRESENT attributes actually support that persona's claims
5. generate_user_story      -> narrative brief for the chosen persona
6. generate_product_content -> agent-optimized copy per product, seeded by the story
7. readiness_score          -> 30% attribute completeness + 70% persona rating (selected persona)
"""

from llm_utils import call_llm_json, call_llm_text

BATCH_SIZE = 20
ATTR_WEIGHT = 0.30
PERSONA_WEIGHT = 0.70


def _s(val, default: str = "") -> str:
    """Safely coerce any field (incl. NaN floats from pandas, None) to a string."""
    if val is None:
        return default
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return default
    except TypeError:
        pass
    return str(val)


# ---------------------------------------------------------------------------
# 1. CLUSTERING
# ---------------------------------------------------------------------------

CLUSTER_SYSTEM_PROMPT = """You group products into meaningful category clusters based on
semantic similarity (what the product IS and who it's for) — not just exact keyword matches.
Return strict JSON: {"clusters": [{"cluster_name": "...", "product_indices": [0,2,5]}, ...]}
Every product index (0-based, as given) must appear in exactly one cluster.
Use short, human-readable cluster names (e.g. "Running Shoes", "Oily Skin Skincare")."""


def _cluster_batch(products_batch: list[dict], offset: int) -> list[dict]:
    listing = "\n".join(
        f"{i}: {_s(p.get('name'), 'Unnamed product')} | "
        f"desc: {_s(p.get('description'))[:150]} | specs: {p.get('specs') or {}}"
        for i, p in enumerate(products_batch)
    )
    result = call_llm_json(
        CLUSTER_SYSTEM_PROMPT,
        f"Group these {len(products_batch)} products into clusters:\n\n{listing}",
    )
    clusters = result.get("clusters", []) if isinstance(result, dict) else []
    # re-offset indices back into the full product list
    for c in clusters:
        c["product_indices"] = [i + offset for i in c.get("product_indices", [])]
    return clusters


MERGE_SYSTEM_PROMPT = """You are given a list of cluster labels produced independently across
several batches of the same catalog. Merge labels that clearly refer to the same category
(e.g. "Running Shoe" and "Running Shoes" are the same). Return strict JSON:
{"merged": [{"final_name": "...", "source_names": ["...", "..."]}, ...]}
Every input label must appear under exactly one final_name."""


def cluster_products(products: list[dict]) -> list[dict]:
    """Returns [{cluster_name, product_indices: [...]}], batching in chunks of BATCH_SIZE."""
    batches = [products[i:i + BATCH_SIZE] for i in range(0, len(products), BATCH_SIZE)]
    raw_clusters = []
    for b_idx, batch in enumerate(batches):
        raw_clusters.extend(_cluster_batch(batch, offset=b_idx * BATCH_SIZE))

    if len(batches) <= 1:
        return raw_clusters

    # Multiple batches -> merge similarly-named clusters produced independently per batch.
    labels = [c["cluster_name"] for c in raw_clusters]
    merge_result = call_llm_json(
        MERGE_SYSTEM_PROMPT,
        f"Cluster labels to merge:\n{labels}",
    )
    merged_groups = merge_result.get("merged", []) if isinstance(merge_result, dict) else []
    if not merged_groups:
        return raw_clusters

    final_clusters = []
    for group in merged_groups:
        indices = []
        for c in raw_clusters:
            if c["cluster_name"] in group.get("source_names", []):
                indices.extend(c["product_indices"])
        final_clusters.append({"cluster_name": group["final_name"], "product_indices": sorted(set(indices))})
    return final_clusters


# ---------------------------------------------------------------------------
# 2 & 3. EXPECTED ATTRIBUTES + COMPLETENESS
# ---------------------------------------------------------------------------

EXPECTED_ATTRS_SYSTEM_PROMPT = """Given a product category and a sample of real products in it,
list the attributes a well-informed buyer (or an AI shopping assistant reasoning on their behalf)
would expect to know about products in this category. Pick 6-10 attributes that are genuinely
decision-relevant for this category (not generic filler).
Return strict JSON: {"expected_attributes": ["attr1", "attr2", ...]}"""


def determine_expected_attributes(cluster_name: str, sample_products: list[dict]) -> list[str]:
    sample_desc = "\n".join(
        f"- {_s(p.get('name'), 'Unnamed product')}: specs={p.get('specs') or {}}"
        for p in sample_products[:8]
    )
    result = call_llm_json(
        EXPECTED_ATTRS_SYSTEM_PROMPT,
        f"Category: {cluster_name}\n\nSample products:\n{sample_desc}",
    )
    return result.get("expected_attributes", []) if isinstance(result, dict) else []


def attribute_completeness(products_in_cluster: list[dict], expected_attributes: list[str]):
    """Returns (cluster_avg_pct, per_product list, missing_attrs_summary dict{attr: count_missing})."""
    if not expected_attributes:
        return 0.0, [], {}

    per_product = []
    missing_counts = {attr: 0 for attr in expected_attributes}

    for p in products_in_cluster:
        specs = p.get("specs") or {}
        spec_keys_lower = {str(k).lower() for k in specs.keys()}
        desc_lower = _s(p.get("description")).lower()
        present = []
        for attr in expected_attributes:
            attr_lower = attr.lower()
            found = any(attr_lower in k or k in attr_lower for k in spec_keys_lower) or attr_lower in desc_lower
            if found:
                present.append(attr)
            else:
                missing_counts[attr] += 1
        pct = round(100 * len(present) / len(expected_attributes), 1)
        per_product.append({"name": _s(p.get("name"), "Unnamed product"), "present": present,
                             "missing": [a for a in expected_attributes if a not in present],
                             "completeness_pct": pct})

    cluster_avg = round(sum(p["completeness_pct"] for p in per_product) / len(per_product), 1) if per_product else 0.0
    return cluster_avg, per_product, missing_counts


# ---------------------------------------------------------------------------
# 4. PERSONA SUGGESTIONS + RATING
# ---------------------------------------------------------------------------

PERSONA_SYSTEM_PROMPT = """You suggest realistic buyer personas / shopping intents for a product
cluster, in the style of natural-language questions people ask AI shopping assistants
(not generic marketing personas).

For each persona, also list which of the GIVEN expected attributes would need to be present
in the product data to credibly support content written for that persona (only choose from the
provided attribute list; pick the 2-5 attributes most central to that persona's decision).

Return strict JSON:
{"personas": [
  {"title": "short persona label", "narrative_seed": "one sentence describing their situation/need",
   "supporting_attributes": ["attr from the given list", ...]},
  ... 2 or 3 total ...
]}"""


def suggest_personas(cluster_name: str, expected_attributes: list[str], present_attrs_by_product: list[dict]):
    # aggregate which expected attributes are present ANYWHERE in the cluster (for context)
    present_anywhere = set()
    for p in present_attrs_by_product:
        present_anywhere.update(p["present"])

    result = call_llm_json(
        PERSONA_SYSTEM_PROMPT,
        f"Cluster: {cluster_name}\n"
        f"Expected attributes for this category: {expected_attributes}\n"
        f"Attributes currently present somewhere in this cluster's data: {sorted(present_anywhere)}",
    )
    personas = result.get("personas", []) if isinstance(result, dict) else []

    for persona in personas:
        supporting = persona.get("supporting_attributes", [])
        # keep only attrs that are actually in our expected list (guard against hallucinated attr names)
        supporting = [a for a in supporting if a in expected_attributes]
        persona["supporting_attributes"] = supporting
        if supporting:
            covered = [a for a in supporting if a in present_anywhere]
            persona["persona_rating_pct"] = round(100 * len(covered) / len(supporting), 1)
            persona["covered_attributes"] = covered
            persona["missing_attributes"] = [a for a in supporting if a not in covered]
        else:
            persona["persona_rating_pct"] = 0.0
            persona["covered_attributes"] = []
            persona["missing_attributes"] = []

    return personas


# ---------------------------------------------------------------------------
# 5. USER STORY GENERATION
# ---------------------------------------------------------------------------

STORY_SYSTEM_PROMPT = """Write ONE short user story in classic "As a ___, I need ___ so that ___"
format, based on the given persona, cluster, and the attributes actually available in the data
(don't invent claims about attributes not listed as available).
Return plain text only, 1-2 sentences, no preamble."""


def generate_user_story(cluster_name: str, persona: dict) -> str:
    prompt = (
        f"Cluster: {cluster_name}\n"
        f"Persona: {persona['title']} — {persona.get('narrative_seed','')}\n"
        f"Attributes confirmed available in the data: {persona.get('covered_attributes', [])}"
    )
    return call_llm_text(STORY_SYSTEM_PROMPT, prompt, temperature=0.7).strip()


# ---------------------------------------------------------------------------
# 6. AGENT-OPTIMIZED CONTENT GENERATION
# ---------------------------------------------------------------------------

CONTENT_SYSTEM_PROMPT = """You write agent-optimized product content: copy meant to be read and
cited by AI shopping assistants answering a specific natural-language buyer intent, NOT
traditional SEO/marketing copy.

Ground every claim in the product's actual name/description/specs given to you. Never invent
numbers, ratings, or specs that were not provided.

Write, in this order, as plain text with clear line breaks (no markdown headers):
1. A 2-3 sentence semantic passage answering the persona's need directly, referencing only
   real attributes given.
2. "Best for:" one line naming the scenario/persona this product suits.
3. One short FAQ-style Q&A addressing a likely objection or follow-up question for this persona.

Keep it concise, concrete, and free of generic marketing fluff ("premium quality", "amazing")."""


def generate_product_content(product: dict, cluster_name: str, persona: dict, user_story: str) -> str:
    prompt = (
        f"Product: {_s(product.get('name'), 'Unnamed product')}\n"
        f"Price: {_s(product.get('price'), 'N/A')}\n"
        f"Description on file: {_s(product.get('description'))}\n"
        f"Specs on file: {product.get('specs') or {}}\n\n"
        f"Cluster/category: {cluster_name}\n"
        f"Target persona: {persona['title']} — {persona.get('narrative_seed','')}\n"
        f"User story: {user_story}"
    )
    return call_llm_text(CONTENT_SYSTEM_PROMPT, prompt, temperature=0.6).strip()


# ---------------------------------------------------------------------------
# 7. READINESS SCORE
# ---------------------------------------------------------------------------

def readiness_score(attribute_completeness_pct: float, persona_rating_pct: float | None) -> float:
    """30% attribute completeness + 70% persona rating of the SELECTED persona.
    Before a persona is selected, persona_rating_pct is None -> treated as 0."""
    pr = persona_rating_pct or 0.0
    return round(attribute_completeness_pct * ATTR_WEIGHT + pr * PERSONA_WEIGHT, 1)
