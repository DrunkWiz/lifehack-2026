"""
Core logic, in the order the app calls it:

1. cluster_products         -> group similar products, batched in chunks of 20
2. determine_expected_attrs -> per cluster, what attributes SHOULD exist for this category
3. attribute_completeness   -> per cluster, % of expected attributes actually present
4. suggest_personas         -> 2-3 candidate personas per cluster, each rated on how well
                                the PRESENT attributes actually support that persona's claims
5. generate_user_story      -> narrative brief for the chosen persona
6. generate_product_content -> agent-optimized copy per product, seeded by the story
7. readiness_score          -> content-fixable coverage only (schema completeness + persona
                                criteria coverage). Fit is reported separately, never blended in.
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
        normalized = p.get("specs_normalized") or {}
        specs = p.get("specs") or {}
        spec_keys_lower = {str(k).lower() for k in specs.keys()}
        desc_lower = _s(p.get("description")).lower()
        present = []
        for attr in expected_attributes:
            if normalized:
                # Normalized path: the schema fixed the key names, so this is an exact
                # lookup rather than substring guesswork.
                value = normalized.get(attr)
                found = value is not None and _s(value).strip() != ""
            else:
                # FALLBACK 4: product was never normalized (schema call failed, batch failed,
                # or every value was rejected by verification) - use the original heuristic.
                attr_lower = attr.lower()
                found = (any(attr_lower in k or k in attr_lower for k in spec_keys_lower)
                         or attr_lower in desc_lower)
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

        if supporting and present_attrs_by_product:
            # Rated PER PRODUCT, then averaged. The previous version scored against the union
            # of attributes present anywhere in the cluster, so one product carrying an
            # attribute credited every other product with it - which inflated the number that
            # drove 70% of the readiness score.
            ratios = []
            for p in present_attrs_by_product:
                product_present = set(p["present"])
                ratios.append(len([a for a in supporting if a in product_present]) / len(supporting))
            persona["persona_rating_pct"] = round(100 * sum(ratios) / len(ratios), 1)
            # kept for display: what the cluster can support at all, vs never supplied by anyone
            persona["covered_attributes"] = [a for a in supporting if a in present_anywhere]
            persona["missing_attributes"] = [a for a in supporting if a not in present_anywhere]
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

Ground every claim in the target product data and peer-product context given to you. You may
derive useful suitability or limitation claims from those facts, but make the reasoning explicit
(for example, low weight + high ventilation -> suited to hot-weather training). Never invent
numbers, ratings, specs, or comparisons. A comparison is allowed only when the relevant target
and peer values are present; name the attribute or scope that proves it. Treat the supplied peers
as the comparison set, not the whole market. Do not call something "best" or "better" without
evidence.

Write, in this order, as plain text with clear line breaks (no markdown headers):
1. A 2-3 sentence semantic passage answering the persona's need directly. Include a grounded
   derived suitability claim and, when peer data supports it, one concrete cluster-relative
   comparison.
2. "Best for:" one line naming the scenario/persona this product suits.
3. "Not for:" one line naming a limitation that follows from the supplied data. If no limitation
   can be supported, say "Not for: No specific limitation established by the available data."
4. Three short FAQ-style Q&As covering different query angles such as fit/use case, conditions,
   price/value, durability, or comparison. Only cover angles supported by the supplied data.

Keep it concise, concrete, and free of generic marketing fluff ("premium quality", "amazing")."""


def _generation_attributes(product: dict) -> dict:
    """Prefer the normalized knowledge layer over raw catalog fields.

    Raw specs on a real export are mostly noise - Handle, Published, Variant SKU, Image Src,
    SEO Title - plus whatever delimited blob the brand's attributes were buried in. Feeding
    that to the writer produces vaguer copy and invites it to treat junk as a product fact.
    Normalized attributes are clean, verified against the source, and consistently named."""
    normalized = product.get("specs_normalized") or {}
    if normalized:
        return {k: v for k, v in normalized.items() if v is not None and _s(v).strip()}
    return product.get("specs") or {}


def _peer_context(product: dict, cluster_products: list[dict] | None) -> list[dict]:
    """Return compact, verified records for other products in the target's cluster."""
    peers = []
    target_name = _s(product.get("name"), "Unnamed product")
    for peer in cluster_products or []:
        peer_name = _s(peer.get("name"), "Unnamed product")
        if peer is product or peer_name == target_name:
            continue
        record = {
            "name": peer_name,
            "price": _s(peer.get("price"), "N/A"),
            "verified_attributes": _generation_attributes(peer),
        }
        # Empty peer records cannot substantiate a comparison and only waste context.
        if record["price"] != "N/A" or record["verified_attributes"]:
            peers.append(record)
    return peers


def generate_product_content(product: dict, cluster_name: str, persona: dict, user_story: str,
                             cluster_products: list[dict] | None = None) -> str:
    peers = _peer_context(product, cluster_products)
    prompt = (
        f"Product: {_s(product.get('name'), 'Unnamed product')}\n"
        f"Price: {_s(product.get('price'), 'N/A')}\n"
        f"Description on file: {_s(product.get('description'))}\n"
        f"Verified attributes: {_generation_attributes(product)}\n\n"
        f"Cluster/category: {cluster_name}\n"
        f"Other products in this cluster (the complete permitted comparison set): {peers}\n"
        f"Comparison-set size: {len(peers) + 1} products including the target\n"
        f"Target persona: {persona['title']} — {persona.get('narrative_seed','')}\n"
        f"User story: {user_story}"
    )
    return call_llm_text(CONTENT_SYSTEM_PROMPT, prompt, temperature=0.6).strip()


# ---------------------------------------------------------------------------
# 7. READINESS SCORE
# ---------------------------------------------------------------------------

def readiness_score(attribute_completeness_pct: float, persona_coverage_pct: float | None) -> float:
    """Readiness measures ONLY what content work can fix: can an agent answer the questions
    this catalog will be asked?

    Two inputs, both coverage: how complete the category schema is, and how much of the
    selected persona's specific criteria the data can answer. They are averaged, not weighted -
    there is no arbitrary split left to justify.

    Fit (does the product actually suit the shopper) is deliberately NOT in here. No amount of
    rewriting makes a 340g shoe lightweight, so folding fit into a content-readiness score
    would penalise brands for a merchandising fact and make the number unactionable. Fit is
    reported alongside instead."""
    if persona_coverage_pct is None:
        return round(attribute_completeness_pct, 1)
    return round((attribute_completeness_pct + persona_coverage_pct) / 2, 1)
