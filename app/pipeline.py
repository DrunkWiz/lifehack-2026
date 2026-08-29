"""
Core logic, in the order the app calls it:

1. cluster_products         -> group similar products, batched in chunks of 20
2. determine_expected_attrs -> per cluster, what attributes SHOULD exist for this category
3. attribute_completeness   -> per cluster, % of expected attributes actually present
4. suggest_personas         -> 2-3 candidate personas per cluster (for the brand to pick from),
                                each rated on how well PRESENT attributes support it
5. generate_user_story      -> narrative brief for the chosen persona
6. generate_agent_content   -> ONE call per product, seeded by the chosen persona/story, that
                                returns a richer structured bundle: personas (incl. a required
                                poor-fit one), not_for (negative info), use_cases grounded in
                                real specs, comparisons against real cluster siblings with a
                                mandatory tradeoff, a narrative passage, and lightweight
                                provenance (catalog_spec vs inferred + unsupported_claims).
7. render_passage           -> assembles the copy-pastable text from the structured bundle,
                                no further model call, so nothing new can be invented here.
8. readiness_score          -> 5 weighted components computed deterministically over what was
                                actually generated (0 for any component before generation runs).
9. ask_confidence            -> given a shopper query + one product's generated content, asks
                                whether the model would recommend it, with a confidence score.
"""

import re
from llm_utils import call_llm_json, call_llm_text

BATCH_SIZE = 20

# ---------------------------------------------------------------------------
# Readiness score weights. Deterministic formula, not a model grading a model.
# Components other than attribute_completeness are 0 until generation has run
# for that product, which is what gives the Before/After view its "before".
# ---------------------------------------------------------------------------
WEIGHTS = {
    "attribute_completeness": 0.25,   # expected attributes actually present in the catalog data
    "persona_coverage": 0.20,         # distinct personas, penalised if none is a "poor" fit
    "not_for_coverage": 0.15,         # negative information — the field marketing copy never writes
    "comparative_context": 0.15,      # comparisons against real siblings that state a tradeoff
    "claim_grounding": 0.25,          # share of fields traced to catalog data vs. inferred
}

LABELS = {
    "attribute_completeness": "Filterable attributes",
    "persona_coverage": "Persona coverage",
    "not_for_coverage": "Negative information",
    "comparative_context": "Comparative context",
    "claim_grounding": "Claim grounding",
}


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


def _parse_price(price) -> float | None:
    if price is None:
        return None
    s = re.sub(r"[^0-9.]", "", str(price))
    try:
        return float(s) if s else None
    except ValueError:
        return None


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

    labels = [c["cluster_name"] for c in raw_clusters]
    merge_result = call_llm_json(MERGE_SYSTEM_PROMPT, f"Cluster labels to merge:\n{labels}")
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
# 4. PERSONA CANDIDATE SUGGESTIONS (for the brand to pick from, per cluster)
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
        supporting = [a for a in persona.get("supporting_attributes", []) if a in expected_attributes]
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
# 6. COMPETITORS (for grounded comparisons) + RICH AGENT CONTENT GENERATION
# ---------------------------------------------------------------------------

def competitors_for(product: dict, cluster_members: list[dict], n: int = 3) -> list[dict]:
    """Closest-priced other products in the same cluster — real siblings, not invented ones."""
    price = _parse_price(product.get("price"))
    others = [p for p in cluster_members if p.get("name") != product.get("name")]
    if price is None:
        return others[:n]
    def _distance(p):
        pp = _parse_price(p.get("price"))
        return abs(pp - price) if pp is not None else float("inf")
    others.sort(key=_distance)
    return others[:n]


AGENT_CONTENT_SYSTEM_PROMPT = """You convert one product's catalog data into a structured bundle
an AI shopping assistant can reason over and a brand can publish. Ground every claim in the data
given to you — never invent numbers, ratings, or specs that were not provided.

RULES

1. GROUNDING. If a spec is absent, omit any claim that depends on it. An omitted detail is
   better than a wrong one.
2. PROVENANCE. For each notable field you write, note whether it is "catalog_spec" (stated in
   the input) or "inferred" (reasoned from general category knowledge, not stated). Anything
   inferred that a shopper could act on and be wrong about goes in unsupported_claims too.
3. NEGATIVE INFORMATION IS MANDATORY. Produce at least two not_for entries — who this product is
   a poor fit for. Be specific ("runners with wide feet"), not vague ("some people").
4. USE CASES MUST CITE SPECS. Every why_it_fits should reference a real spec value, not a vague
   adjective.
5. COMPARISONS need a tradeoff. Compare only against the real competitor products given to you.
   "Lighter, but less cushioned" is usable; "better overall" is not.
6. PERSONAS must include AT LEAST ONE with fit="poor". A bundle where every persona is a strong
   fit helps rank nothing — that is a required field, not optional.

Return strict JSON:
{
  "personas": [{"label": "...", "fit": "strong|partial|poor", "reasoning": "..."}, ...],
  "not_for": [{"exclusion": "...", "reason": "..."}, ...],
  "use_cases": [{"scenario": "...", "why_it_fits": "...", "grounded_in": ["spec key", ...]}, ...],
  "comparisons": [{"against": "competitor product name given to you", "axis": "...",
                    "direction": "more|less|similar", "tradeoff": "..."}, ...],
  "narrative": {"one_line_pitch": "...", "best_for": "...", "faq_question": "...", "faq_answer": "..."},
  "field_sources": [{"field": "short label", "source": "catalog_spec|inferred"}, ...],
  "unsupported_claims": ["..."]
}"""


def generate_agent_content(product: dict, cluster_name: str, persona: dict, user_story: str,
                            competitors: list[dict]) -> dict:
    comp_lines = [
        f"- {_s(c.get('name'), 'Unnamed')} ({_s(c.get('price'), 'price unknown')}): "
        f"{c.get('specs') or {}}"
        for c in competitors
    ] or ["- (no other products in this cluster to compare against)"]

    prompt = (
        f"PRODUCT\n"
        f"Name: {_s(product.get('name'), 'Unnamed product')}\n"
        f"Price: {_s(product.get('price'), 'N/A')}\n"
        f"Description: {_s(product.get('description'))}\n"
        f"Specs: {product.get('specs') or {}}\n\n"
        f"CATEGORY/CLUSTER: {cluster_name}\n"
        f"SEED PERSONA: {persona.get('title','')} — {persona.get('narrative_seed','')}\n"
        f"SEED USER STORY: {user_story}\n\n"
        f"COMPETITOR CONTEXT (real siblings in this cluster):\n" + "\n".join(comp_lines)
    )
    result = call_llm_json(AGENT_CONTENT_SYSTEM_PROMPT, prompt, temperature=0.4)
    if not isinstance(result, dict):
        result = {}
    # Coerce, don't just default-if-missing: the model can return a key with an
    # explicit null value, which .setdefault() would not catch.
    result["personas"] = result.get("personas") or []
    result["not_for"] = result.get("not_for") or []
    result["use_cases"] = result.get("use_cases") or []
    result["comparisons"] = result.get("comparisons") or []
    result["narrative"] = result.get("narrative") or {}
    result["field_sources"] = result.get("field_sources") or []
    result["unsupported_claims"] = result.get("unsupported_claims") or []
    return result


# ---------------------------------------------------------------------------
# 7. RENDER — pure formatting, no further model call
# ---------------------------------------------------------------------------

def build_raw_passage(product: dict) -> str:
    """Deterministic, no LLM call: what the catalog says today, formatted the same
    shape as render_passage() so the Ask tab can test raw content before any
    generation has run — this is the 'before' half of a live demo."""
    lines = [_s(product.get("name"), "Unnamed product")]
    price = _s(product.get("price"))
    if price:
        lines.append(f"Price: {price}")
    desc = _s(product.get("description"))
    if desc:
        lines.append(desc)
    specs = product.get("specs") or {}
    if specs:
        lines.append("Specs: " + ", ".join(f"{k}: {v}" for k, v in specs.items()))
    return "\n".join(lines).strip()


def render_passage(agent_content: dict) -> str:
    n = agent_content.get("narrative", {}) or {}
    lines = []
    if n.get("one_line_pitch"):
        lines.append(n["one_line_pitch"])
    if n.get("best_for"):
        lines.append(f"\nBest for: {n['best_for']}")
    if n.get("faq_question") and n.get("faq_answer"):
        lines.append(f"\nQ: {n['faq_question']}\nA: {n['faq_answer']}")

    not_for = agent_content.get("not_for") or []
    if not_for:
        lines.append("\nNot a fit for:")
        for item in not_for:
            lines.append(f"- {item.get('exclusion','')}: {item.get('reason','')}")

    comparisons = agent_content.get("comparisons") or []
    if comparisons:
        lines.append("\nHow it compares:")
        for c in comparisons:
            lines.append(
                f"- vs. {c.get('against','')}: {c.get('direction','')} {c.get('axis','')} "
                f"— {c.get('tradeoff','')}"
            )

    unsupported = agent_content.get("unsupported_claims") or []
    if unsupported:
        lines.append("\n[Needs human review before publishing]")
        for u in unsupported:
            lines.append(f"- {u}")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 8. READINESS SCORE — deterministic, 5 weighted components
# ---------------------------------------------------------------------------

def score_components(attribute_completeness_pct: float, agent_content: dict | None) -> dict:
    attr = round((attribute_completeness_pct or 0.0) / 100, 4)

    if not agent_content:
        return {"attribute_completeness": attr, "persona_coverage": 0.0,
                "not_for_coverage": 0.0, "comparative_context": 0.0, "claim_grounding": 0.0}

    personas = agent_content.get("personas") or []
    labels = {p.get("label", "").strip().lower() for p in personas if p.get("label")}
    pers = min(len(labels), 4) / 4
    has_poor_fit = any(p.get("fit") == "poor" for p in personas)
    if not has_poor_fit or not personas:
        pers *= 0.7  # penalise all-positive bundles

    not_for = agent_content.get("not_for") or []
    not_for_score = min(len(not_for), 2) / 2

    comparisons = agent_content.get("comparisons") or []
    comp = min(sum(1 for c in comparisons if c.get("tradeoff")), 3) / 3

    sources = agent_content.get("field_sources") or []
    grounded = sum(1 for s in sources if s.get("source") == "catalog_spec")
    ground = grounded / max(len(sources), 1) if sources else 0.5  # neutral if nothing tagged
    unsupported = agent_content.get("unsupported_claims") or []
    ground *= max(0.0, 1 - 0.1 * len(unsupported))

    return {
        "attribute_completeness": round(attr, 4),
        "persona_coverage": round(pers, 4),
        "not_for_coverage": round(not_for_score, 4),
        "comparative_context": round(comp, 4),
        "claim_grounding": round(max(ground, 0.0), 4),
    }


def readiness_score(attribute_completeness_pct: float, agent_content: dict | None = None) -> float:
    parts = score_components(attribute_completeness_pct, agent_content)
    return round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()), 1)


def top_gaps(attribute_completeness_pct: float, agent_content: dict | None, missing_attrs: list[str]) -> list[str]:
    gaps = []
    if missing_attrs:
        gaps.append("Missing filterable attributes: " + ", ".join(missing_attrs))
    if not agent_content:
        gaps.append("No persona selected yet — pick one to generate content")
        return gaps[:5]

    personas = agent_content.get("personas") or []
    if not any(p.get("fit") == "poor" for p in personas):
        gaps.append("No 'poor fit' persona — a card that suits everyone ranks for no one")
    if not agent_content.get("not_for"):
        gaps.append("No negative information — nothing stops a wrong recommendation")
    if not any(c.get("tradeoff") for c in (agent_content.get("comparisons") or [])):
        gaps.append("No comparison states a tradeoff, so agents can't rank this against siblings")
    if agent_content.get("unsupported_claims"):
        gaps.append(f"{len(agent_content['unsupported_claims'])} claim(s) flagged unsupported and pending review")
    gaps.sort(key=lambda g: -len(g))
    return gaps[:5]


# ---------------------------------------------------------------------------
# 9. ASK — one query against one product's generated content
# ---------------------------------------------------------------------------

ASK_SYSTEM_PROMPT = """You are an AI shopping assistant deciding whether to recommend a
product for a shopper's query, based ONLY on the product content given to you.
Return strict JSON: {"recommend": true|false, "confidence": 0-100, "reason": "one sentence"}
confidence reflects how well-supported your decision is by the content, not how good the
product sounds — a vague passage should get a low confidence even if it sounds positive."""


def ask_confidence(query: str, product_name: str, passage_text: str) -> dict:
    prompt = f"Shopper query: {query}\n\nProduct: {product_name}\n\nProduct content:\n{passage_text}"
    result = call_llm_json(ASK_SYSTEM_PROMPT, prompt, temperature=0.2)
    if not isinstance(result, dict):
        result = {}
    result.setdefault("recommend", False)
    result.setdefault("confidence", 0)
    result.setdefault("reason", "")
    try:
        result["confidence"] = max(0, min(100, int(result["confidence"])))
    except (TypeError, ValueError):
        result["confidence"] = 0
    return result
