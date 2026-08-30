"""
Core logic, in the order the app calls it:

1. cluster_products         -> group similar products, batched in chunks of 20
2. determine_expected_attrs -> per cluster, what attributes SHOULD exist for this category
3. attribute_completeness   -> per cluster, % of expected attributes actually present
4. suggest_personas         -> 2-3 candidate personas per cluster, each rated on how well
                                the PRESENT attributes actually support that persona's claims
5. generate_user_story      -> narrative brief for the chosen persona
6. generate_agent_content   -> structured, agent-optimized content seeded by the story
7. readiness_score          -> content-fixable coverage only (schema completeness + persona
                                criteria coverage). Fit is reported separately, never blended in.
"""

import re

from llm_utils import call_llm_json, call_llm_text

BATCH_SIZE = 20
AGENT_CONTENT_SCHEMA_VERSION = 3
# Deterministic readiness-score weights. These sum to 1.0.
WEIGHTS = {
    "attribute_completeness": 0.25,
    "persona_coverage": 0.20,
    "not_for_coverage": 0.15,
    "comparative_context": 0.15,
    "claim_grounding": 0.25,
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


def _parse_price(value) -> float | None:
    """Extract the first numeric price from values such as 'S$189' or '$1,299.00'."""
    text = _s(value).strip()
    if not text:
        return None
    match = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
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
    # re-offset indices back into the full product list
    for c in clusters:
        c["product_indices"] = [i + offset for i in c.get("product_indices", [])]
    return clusters


def _validate_cluster_membership(clusters: list[dict], product_count: int) -> list[dict]:
    """Sanitize model output so every catalog index appears exactly once."""
    clean = []
    assigned = set()
    for position, cluster in enumerate(clusters or []):
        if not isinstance(cluster, dict):
            continue
        name = _s(cluster.get("cluster_name")).strip() or f"Cluster {position + 1}"
        indices = []
        for value in cluster.get("product_indices") or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= index < product_count and index not in assigned:
                assigned.add(index)
                indices.append(index)
        if indices:
            clean.append({"cluster_name": name, "product_indices": sorted(indices)})

    missing = [index for index in range(product_count) if index not in assigned]
    if missing:
        clean.append({"cluster_name": "Uncategorized", "product_indices": missing})
    return clean


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
        return _validate_cluster_membership(raw_clusters, len(products))

    # Multiple batches -> merge similarly-named clusters produced independently per batch.
    labels = [c["cluster_name"] for c in raw_clusters]
    merge_result = call_llm_json(
        MERGE_SYSTEM_PROMPT,
        f"Cluster labels to merge:\n{labels}",
    )
    merged_groups = merge_result.get("merged", []) if isinstance(merge_result, dict) else []
    if not merged_groups:
        return _validate_cluster_membership(raw_clusters, len(products))

    final_clusters = []
    for group in merged_groups:
        indices = []
        for c in raw_clusters:
            if c["cluster_name"] in group.get("source_names", []):
                indices.extend(c["product_indices"])
        final_clusters.append({"cluster_name": group["final_name"], "product_indices": sorted(set(indices))})
    return _validate_cluster_membership(final_clusters, len(products))


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
        applicable = ([a for a in expected_attributes if a in p["applicable_attributes"]]
                      if "applicable_attributes" in p else list(expected_attributes))
        applicable_set = set(applicable)
        spec_keys_lower = {str(k).lower() for k in specs.keys()}
        desc_lower = _s(p.get("description")).lower()
        present = []
        for attr in expected_attributes:
            if attr not in applicable_set:
                continue
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
        pct = round(100 * len(present) / len(applicable), 1) if applicable else 100.0
        per_product.append({"product_id": _s(p.get("product_id")),
                             "name": _s(p.get("name"), "Unnamed product"), "present": present,
                             "missing": [a for a in applicable if a not in present],
                             "not_applicable": [a for a in expected_attributes if a not in applicable_set],
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

        if supporting and present_attrs_by_product:
            # Rated PER PRODUCT, then averaged. The previous version scored against the union
            # of attributes present anywhere in the cluster, so one product carrying an
            # attribute credited every other product with it and inflated persona coverage.
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
# 6. COMPETITORS (for grounded comparisons) + RICH AGENT CONTENT GENERATION
# ---------------------------------------------------------------------------

def competitors_for(product: dict, cluster_members: list[dict], n: int = 3) -> list[dict]:
    """Closest-priced other products in the same cluster — real siblings, not invented ones."""
    price = _parse_price(product.get("price"))
    product_id = product.get("product_id")
    others = ([p for p in cluster_members if p.get("product_id") != product_id]
              if product_id else [p for p in cluster_members if p is not product])
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
   better than a wrong one. The persona and user story describe the shopper, not evidence about
   the product; never turn them into product claims.
2. PROVENANCE. For each notable field you write, note whether it is "catalog_spec" (stated in
   the input) or "inferred" (reasoned from the supplied evidence). Evidence-backed inferences
   belong in derived_insights. Put a claim in unsupported_claims when the supplied attributes do
   not support it; never present an inference as a catalog fact.
3. DERIVED INSIGHTS bridge catalog language to shopper intent. Produce 3-5 useful suitability,
   benefit, or tradeoff inferences across distinct decision angles. Each must show its reasoning
   and cite exact attribute/value evidence. Label these as derived insights, never verified facts.
4. QUERY ANGLES. Produce 3-4 short questions shoppers could ask from different relevant angles
   such as use case, environment, fit, routine, performance, budget, materials, or limitations.
   Answer only what the evidence supports; do not force irrelevant angles.
5. NEGATIVE INFORMATION IS MANDATORY. Produce at least two not_for entries. Each exclusion must
   cite exact attribute/value evidence. Never infer the absence of a feature from an unrelated
   category label.
6. USE CASES MUST CITE SPECS. Every why_it_fits should reference exact attribute/value evidence.
7. COMPARISONS need a tradeoff. Compare only attributes supplied for both products. Copy the
   exact target_value and competitor_value provided. Unverifiable comparisons will be removed.
8. PERSONAS must include AT LEAST ONE with fit="poor". A bundle where every persona is a strong
   fit helps rank nothing — that is a required field, not optional.

Return strict JSON:
{
  "personas": [{"label": "...", "fit": "strong|partial|poor", "reasoning": "..."}, ...],
  "derived_insights": [{"angle": "...", "claim": "...", "reasoning": "...",
                         "evidence": [{"attribute": "exact key", "value": "exact value"}],
                         "confidence": "high|medium"}, ...],
  "query_angles": [{"angle": "...", "shopper_question": "...", "answer": "...",
                     "evidence": [{"attribute": "exact key", "value": "exact value"}]}, ...],
  "not_for": [{"exclusion": "...", "reason": "...",
                "evidence": [{"attribute": "exact key", "value": "exact value"}]}, ...],
  "use_cases": [{"scenario": "...", "why_it_fits": "...",
                  "evidence": [{"attribute": "exact key", "value": "exact value"}]}, ...],
  "comparisons": [{"against": "competitor product name given to you", "axis": "verified key",
                    "direction": "more|less|similar", "target_value": "exact supplied value",
                    "competitor_value": "exact supplied value", "tradeoff": "..."}, ...],
  "narrative": {"one_line_pitch": "...", "best_for": "...", "faq_question": "...", "faq_answer": "..."},
  "field_sources": [{"field": "short label", "source": "catalog_spec|inferred"}, ...],
  "unsupported_claims": ["..."]
}"""


DERIVATION_REVIEW_SYSTEM_PROMPT = """You are a strict evidence-entailment reviewer for product
content. Decide whether each proposed statement is reasonably supported by its cited verified
product facts. Evidence must be relevant to the conclusion, not merely true.

Reject statements that require an unstated product capability or intended use. In particular:
- price and weight do not prove hot/humid-weather suitability;
- a casual or daily-use label does not prove marathon, racing, recovery, or technical suitability;
- a road label does not prove suitability for every road-running distance;
- the absence of an attribute does not prove the product lacks that feature;
- generic category knowledge cannot replace product-specific evidence.

Reasonable direct implications are allowed when explicitly labelled as derived, such as high
ventilation supporting airflow in warm conditions. When uncertain, reject.

Return strict JSON: {"approved_ids": ["field:index", ...],
"rejected": [{"id": "field:index", "reason": "short explanation"}]}"""


def _verified_attributes(product: dict) -> dict:
    verified = {
        str(key): value for key, value in (product.get("specs_normalized") or {}).items()
        if value is not None and _s(value).strip()
    }
    price = _s(product.get("price")).strip()
    if price:
        verified["catalog_price"] = price
    return verified


def _canonical(value) -> str:
    return re.sub(r"[^a-z0-9]", "", _s(value).lower())


def _numeric(value) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", _s(value).replace(",", ""))
    return float(match.group()) if match else None


def _validate_comparisons(product: dict, competitors: list[dict], comparisons: list) -> tuple[list, list]:
    """Keep only comparisons whose product, axis, values, and direction are provable."""
    target = _verified_attributes(product)
    target_keys = {_canonical(key): key for key in target}
    competitor_map = {_canonical(c.get("name")): c for c in competitors}
    valid, rejected = [], []
    for comparison in comparisons or []:
        if not isinstance(comparison, dict):
            continue
        competitor = competitor_map.get(_canonical(comparison.get("against")))
        axis_key = target_keys.get(_canonical(comparison.get("axis")))
        competitor_attrs = _verified_attributes(competitor or {})
        competitor_keys = {_canonical(key): key for key in competitor_attrs}
        competitor_key = competitor_keys.get(_canonical(axis_key)) if axis_key else None
        direction = _s(comparison.get("direction")).lower()
        reason = None
        if not competitor:
            reason = "competitor was not supplied"
        elif not axis_key or not competitor_key:
            reason = "comparison attribute was not verified for both products"
        else:
            target_value = target[axis_key]
            competitor_value = competitor_attrs[competitor_key]
            if _canonical(comparison.get("target_value")) != _canonical(target_value) or \
                    _canonical(comparison.get("competitor_value")) != _canonical(competitor_value):
                reason = "comparison values did not match the verified attributes"
            else:
                left, right = _numeric(target_value), _numeric(competitor_value)
                expected = ("similar" if left == right else "more" if left > right else "less") \
                    if left is not None and right is not None else \
                    ("similar" if _canonical(target_value) == _canonical(competitor_value) else None)
                if expected is None or direction != expected:
                    reason = "comparison direction could not be verified"
        if reason:
            rejected.append(
                f"Comparison against {_s(comparison.get('against'), 'unknown product')} "
                f"on {_s(comparison.get('axis'), 'unknown attribute')} was removed: {reason}."
            )
        else:
            comparison["axis"] = axis_key
            comparison["tradeoff"] = (
                f"Verified {axis_key}: this product is {target[axis_key]}; "
                f"{_s(competitor.get('name'), 'competitor')} is {competitor_attrs[competitor_key]}."
            )
            valid.append(comparison)
    return valid, rejected


def _validate_evidence_items(product: dict, items: list, label: str) -> tuple[list, list]:
    """Require every inference to cite exact keys and values from the verified product facts."""
    verified = _verified_attributes(product)
    keys = {_canonical(key): key for key in verified}
    required_fields = {
        "Derived insight": ("claim", "reasoning"),
        "Query angle": ("shopper_question", "answer"),
        "Exclusion": ("exclusion", "reason"),
        "Use case": ("scenario", "why_it_fits"),
    }.get(label, ())
    valid, rejected = [], []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") or []
        normalized_evidence = []
        reason = None
        if any(not _s(item.get(field)).strip() for field in required_fields):
            reason = "required explanatory text was missing"
        elif not evidence:
            reason = "no attribute evidence was supplied"
        for citation in evidence:
            if not isinstance(citation, dict):
                reason = "evidence was malformed"
                break
            key = keys.get(_canonical(citation.get("attribute")))
            if not key or _canonical(citation.get("value")) != _canonical(verified.get(key)):
                reason = "an evidence key or value did not match verified attributes"
                break
            normalized_evidence.append({"attribute": key, "value": verified[key]})
        if reason:
            item_text = next((_s(item.get(field)) for field in
                              ("claim", "shopper_question", "exclusion", "scenario")
                              if _s(item.get(field)).strip()), "unnamed item")
            rejected.append(f"{label} '{item_text}' was removed: {reason}.")
        else:
            item["evidence"] = normalized_evidence
            valid.append(item)
    return valid, rejected


def _semantic_review(product: dict, fields: dict[str, list]) -> tuple[dict[str, list], list[str]]:
    """Use a separate strict pass to check that true evidence actually entails each inference."""
    review_items = []
    text_fields = {
        "derived_insights": ("claim", "reasoning"),
        "query_angles": ("shopper_question", "answer"),
        "not_for": ("exclusion", "reason"),
        "use_cases": ("scenario", "why_it_fits"),
    }
    for field, items in fields.items():
        for index, item in enumerate(items):
            review_items.append({
                "id": f"{field}:{index}",
                "statement": " — ".join(_s(item.get(key)) for key in text_fields[field]),
                "evidence": item.get("evidence") or [],
            })
    if not review_items:
        return fields, []

    result = call_llm_json(
        DERIVATION_REVIEW_SYSTEM_PROMPT,
        f"Verified product facts: {_verified_attributes(product)}\n\n"
        f"Proposed statements: {review_items}",
        temperature=0.0,
    )
    approved = set(result.get("approved_ids") or []) if isinstance(result, dict) else set()
    rejected_items = (result.get("rejected") or []) if isinstance(result, dict) else []
    rejected_reasons = {
        _s(item.get("id")): _s(item.get("reason"), "semantic support was not established")
        for item in rejected_items if isinstance(item, dict)
    }
    reviewed = {}
    rejected = []
    for field, items in fields.items():
        reviewed[field] = []
        for index, item in enumerate(items):
            item_id = f"{field}:{index}"
            if item_id in approved:
                reviewed[field].append(item)
            else:
                statement = next((_s(item.get(key)) for key in text_fields[field]
                                  if _s(item.get(key)).strip()), "unnamed item")
                rejected.append(
                    f"Derived statement '{statement}' was removed: "
                    f"{rejected_reasons.get(item_id, 'semantic support was not established')}."
                )
    return reviewed, rejected


def generate_agent_content(product: dict, cluster_name: str, persona: dict, user_story: str,
                            competitors: list[dict]) -> dict:
    verified = _verified_attributes(product)
    if not any(_s(value).strip() for value in (product.get("specs_normalized") or {}).values()
               if value is not None):
        return {
            "schema_version": AGENT_CONTENT_SCHEMA_VERSION,
            "personas": [], "derived_insights": [], "query_angles": [],
            "not_for": [], "use_cases": [], "comparisons": [],
            "narrative": {}, "field_sources": [],
            "unsupported_claims": [
                "No agent content was generated because this product has no verified normalized attributes."
            ],
        }
    comp_lines = [
        f"- {_s(c.get('name'), 'Unnamed')}: "
        f"verified_attributes={_verified_attributes(c)}"
        for c in competitors
    ] or ["- (no other products in this cluster to compare against)"]

    prompt = (
        f"PRODUCT\n"
        f"Name: {_s(product.get('name'), 'Unnamed product')}\n"
        f"Verified normalized attributes (the only permitted factual specs): "
        f"{verified}\n\n"
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
    result["derived_insights"] = result.get("derived_insights") or []
    result["query_angles"] = result.get("query_angles") or []
    result["not_for"] = result.get("not_for") or []
    result["use_cases"] = result.get("use_cases") or []
    result["comparisons"] = result.get("comparisons") or []
    result["narrative"] = result.get("narrative") or {}
    result["field_sources"] = result.get("field_sources") or []
    result["unsupported_claims"] = result.get("unsupported_claims") or []
    result["schema_version"] = AGENT_CONTENT_SCHEMA_VERSION
    for field, label in (("derived_insights", "Derived insight"),
                         ("query_angles", "Query angle"),
                         ("not_for", "Exclusion"),
                         ("use_cases", "Use case")):
        result[field], evidence_rejections = _validate_evidence_items(
            product, result[field], label
        )
        result["unsupported_claims"].extend(evidence_rejections)
    reviewed, semantic_rejections = _semantic_review(product, {
        field: result[field] for field in
        ("derived_insights", "query_angles", "not_for", "use_cases")
    })
    result.update(reviewed)
    result["unsupported_claims"].extend(semantic_rejections)
    valid_comparisons, rejected = _validate_comparisons(
        product, competitors, result["comparisons"]
    )
    result["comparisons"] = valid_comparisons
    result["unsupported_claims"].extend(rejected)
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
    lines = []

    insights = agent_content.get("derived_insights") or []
    if insights:
        lines.append("\nDerived product insights:")
        for item in insights:
            evidence = ", ".join(
                f"{e.get('attribute')}: {e.get('value')}" for e in item.get("evidence", [])
            )
            lines.append(
                f"- {item.get('claim','')} Reasoning: {item.get('reasoning','')} "
                f"[Evidence: {evidence}]"
            )

    use_cases = agent_content.get("use_cases") or []
    if use_cases:
        lines.append("\nRelevant use cases:")
        for item in use_cases:
            lines.append(f"- {item.get('scenario','')}: {item.get('why_it_fits','')}")

    query_angles = agent_content.get("query_angles") or []
    if query_angles:
        lines.append("\nQuestions this product can answer:")
        for item in query_angles:
            lines.append(f"- Q: {item.get('shopper_question','')} A: {item.get('answer','')}")

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

    unsupported = agent_content.get("unsupported_claims") or []
    evidence_backed = sum(len(agent_content.get(field) or []) for field in (
        "derived_insights", "query_angles", "use_cases", "not_for", "comparisons"
    ))
    total_claims = evidence_backed + len(unsupported)
    ground = evidence_backed / total_claims if total_claims else 0.5

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
# 9. CATALOG-AWARE DEMO QUERIES
# ---------------------------------------------------------------------------

SUGGESTED_QUERIES_SYSTEM_PROMPT = """Create three realistic questions a shopper might ask an
AI assistant about the supplied catalog. The questions are demo inputs for testing product
recommendations, not questions about the catalog itself.

Rules:
- Use only product categories and decision factors supported by the catalog summary.
- Write natural first-person shopping requests, not keyword searches.
- Each question should combine a task, useful context, at least one hard constraint, and one
  preference so ranking requires reasoning.
- Make the three questions meaningfully different. Include a known catalog gap in at most one
  question so the demo can also expose missing information.
- Do not mention a specific product name or invent exact product facts.
- Keep each question to one sentence.

Return strict JSON: {"queries": ["...", "...", "..."]}"""


def suggest_shopper_queries(cluster_data: dict) -> list[str]:
    summary = []
    for cluster_name, data in cluster_data.items():
        members = data.get("members") or []
        summary.append({
            "category": cluster_name,
            "product_count": len(members),
            "sample_prices": [_s(p.get("price")) for p in members[:8] if _s(p.get("price"))],
            "available_attributes": data.get("expected_attrs") or [],
            "missing_attributes": [key for key, count in (data.get("missing_counts") or {}).items()
                                   if count],
            "shopper_intents": [p.get("narrative_seed") or p.get("title")
                                for p in (data.get("personas") or [])],
        })
    result = call_llm_json(
        SUGGESTED_QUERIES_SYSTEM_PROMPT,
        f"Catalog summary:\n{summary}",
        temperature=0.5,
    )
    queries = result.get("queries", []) if isinstance(result, dict) else []
    return [str(query).strip() for query in queries if str(query).strip()][:3]


# ---------------------------------------------------------------------------
# 10. ASK — one query against one product's generated content
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
