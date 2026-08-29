"""
Retrieval layer — the part that answers "would an AI assistant actually pick this product?"

The rest of the app PRODUCES agent-optimized content. This module TESTS it: a shopper's
natural-language query is run against the catalog twice — once over the raw catalog text
as extracted, once over the generated content — so the two ranked result sets can be shown
side by side.

Deliberately no vector database. At catalog scale (hundreds to low thousands of products)
a dot product over a row-normalized embedding matrix is sub-millisecond, needs no extra
dependency beyond numpy, and keeps the Streamlit Cloud image small and the cold start fast.
faiss/chroma would add install weight and boot latency for no measurable gain here.
"""

import json
import math
import re
import numpy as np

from llm_utils import embed_texts, call_llm_json

TOP_K = 5
CANDIDATE_K = 20  # broad retrieval pool for constraint-aware reranking


def _s(val, default: str = "") -> str:
    """Safely coerce any field (incl. NaN floats from pandas, None) to a string."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return str(val)


# ---------------------------------------------------------------------------
# Text views of a product — the two things we compare
# ---------------------------------------------------------------------------

def product_to_raw_text(product: dict) -> str:
    """The catalog as the brand actually supplied it. This is the 'before'."""
    specs = product.get("specs") or {}
    spec_str = "; ".join(f"{k}: {v}" for k, v in specs.items() if _s(v).strip())
    parts = [
        _s(product.get("name"), "Unnamed product"),
        f"Price: {_s(product.get('price'), 'not stated')}",
        _s(product.get("description")),
        spec_str,
    ]
    return "\n".join(p for p in parts if p.strip())


def product_to_optimized_text(product: dict, generated: str) -> str:
    """Generated content, kept with name and price so both sides can match on identity
    and budget. Without these the comparison would be unfair in the other direction.

    Normalized attributes are appended too: the prose is written for one persona, so a query
    from a different angle ("wide fit", "stability") would otherwise have nothing to match on
    even though the data is right there. This is what stops persona-targeted copy from
    degrading retrieval for every other intent."""
    normalized = product.get("specs_normalized") or {}
    parts = [
        _s(product.get("name"), "Unnamed product"),
        f"Price: {_s(product.get('price'), 'not stated')}",
    ]
    if normalized:
        parts.append("Attributes: " + "; ".join(
            f"{k}: {v}" for k, v in normalized.items() if v is not None and _s(v).strip()))
    # Keep verified facts ahead of persona prose so they survive candidate prompt truncation.
    parts.append(_s(generated))
    return "\n".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Index build + search
# ---------------------------------------------------------------------------

def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_index(entries: list[dict]) -> dict:
    """entries: [{"key": str, "name": str, "cluster": str, "text": str}, ...]

    Returns {"matrix": np.ndarray | None, "entries": [...]}. Rows are L2-normalized,
    so cosine similarity is a plain dot product at query time."""
    entries = [e for e in entries if _s(e.get("text")).strip()]
    if not entries:
        return {"matrix": None, "entries": []}
    vectors = embed_texts([e["text"] for e in entries])
    matrix = _normalize(np.array(vectors, dtype=np.float32))
    return {"matrix": matrix, "entries": entries}


def search(index: dict, query_vec: np.ndarray, k: int = TOP_K) -> list[dict]:
    """Cosine similarity against a normalized matrix. Returns ranked hits."""
    if index.get("matrix") is None or not index.get("entries"):
        return []
    sims = index["matrix"] @ query_vec
    k = min(k, len(sims))
    top = np.argsort(-sims)[:k]
    return [
        {
            "rank": position + 1,
            "score": round(float(sims[row]), 4),
            "key": index["entries"][row]["key"],
            "name": index["entries"][row]["name"],
            "cluster": index["entries"][row].get("cluster", ""),
            "text": index["entries"][row]["text"],
            "attributes": index["entries"][row].get("attributes", {}),
        }
        for position, row in enumerate(top)
    ]


def embed_query(query: str) -> np.ndarray:
    vec = np.array(embed_texts([query])[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# ---------------------------------------------------------------------------
# Intent parsing + rerank
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """Decompose a shopper request without changing its meaning.

Separate:
- task: what the shopper is doing and what product they seek
- context: environmental or situational facts
- constraints: explicit requirements, each marked hard or preference
- derived_needs: product-selection requirements reasonably inferred from the task or context

Rules:
- Preserve action and temporal framing. "I am running a half marathon" means participating in
  that event; it does NOT mean training for one. Never introduce train/training unless stated.
- Preserve a short exact source_text from the request for every extracted item.
- Numeric ceilings, required product types, and explicit exclusions are hard constraints.
- Product descriptors explicitly requested by the shopper (for example "stability shoes") are
  hard constraints, not preferences.
- A phrase requiring a specified attribute without giving a value (for example "must have a
  specific heel-to-toe drop") is a hard `exists` constraint.
- Map every hard constraint to exactly one name from the supplied available attribute names.
- Use only these operators: eq, contains, exists, lt, lte, gt, gte.
- Subjective words without a threshold, such as "lightweight", are relative preferences rather
  than binary pass/fail constraints.
- Derived needs must state what explicit task/context they came from. For example, humid weather
  can imply a need for ventilation. Keep them separate from the shopper's explicit constraints.
- Do not add product claims.

Return strict JSON:
{"task": {"action": "...", "goal": "...", "product_sought": "...", "source_text": "..."},
 "context": [{"factor": "...", "value": "...", "source_text": "..."}],
 "constraints": [{"requirement": "...", "normalized_attribute": "exact available name",
                  "operator": "eq|contains|exists|lt|lte|gt|gte", "value": null,
                  "unit": null, "strength": "hard|preference", "source_text": "..."}],
 "derived_needs": [{"need": "...", "derived_from": "...", "rationale": "..."}],
 "ambiguities": ["..."]}"""


def parse_shopper_intent(query: str, available_attributes: dict[str, list[str]]) -> dict:
    result = call_llm_json(
        INTENT_SYSTEM_PROMPT,
        f"Available exact attributes and sample values: {available_attributes}\n\n"
        f"Shopper request:\n{query}",
        temperature=0.0,
    )
    if not isinstance(result, dict):
        result = {}
    task = result.get("task")
    result["task"] = task if isinstance(task, dict) else {
        "action": "", "goal": query, "product_sought": "", "source_text": query
    }
    result["context"] = result.get("context") if isinstance(result.get("context"), list) else []
    result["constraints"] = (result.get("constraints")
                             if isinstance(result.get("constraints"), list) else [])
    result["derived_needs"] = (result.get("derived_needs")
                               if isinstance(result.get("derived_needs"), list) else [])
    result["ambiguities"] = (result.get("ambiguities")
                             if isinstance(result.get("ambiguities"), list) else [])
    return result


def _number(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", _s(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _constraint_check(constraint: dict, attributes: dict) -> dict:
    requirement = _s(constraint.get("source_text") or constraint.get("requirement"), "Requirement")
    attribute_name = _s(constraint.get("normalized_attribute"))
    operator = _s(constraint.get("operator"), "eq").lower()
    operator = {"=": "eq", "==": "eq", "<": "lt", "<=": "lte",
                ">": "gt", ">=": "gte"}.get(operator, operator)
    expected = constraint.get("value")
    actual = attributes.get(attribute_name)
    resolved_attribute = attribute_name

    # Schemas inferred independently for sibling clusters can call the same concept
    # support, support_type, or support_level. Resolve those variants deterministically.
    if actual is None and attribute_name:
        key = re.sub(r"[^a-z0-9]", "", attribute_name.lower())
        for candidate_key, candidate_value in attributes.items():
            candidate = re.sub(r"[^a-z0-9]", "", candidate_key.lower())
            if key in candidate or candidate in key:
                actual = candidate_value
                resolved_attribute = candidate_key
                break

    # If the parser chose a differently named product-type field, an explicit expected
    # value can still be proven by a verified fact such as category="Stability Running Shoes".
    matched_expected = False
    expected_text = _s(expected).lower().strip()
    if (actual is None or not _s(actual).strip()) and expected_text:
        for candidate_key, candidate_value in attributes.items():
            value_text = _s(candidate_value).lower()
            if expected_text in value_text:
                actual = candidate_value
                resolved_attribute = candidate_key
                matched_expected = True
                break

    if actual is None or not _s(actual).strip():
        return {"requirement": requirement, "attribute": attribute_name,
                "status": "unknown", "actual": None}
    if operator == "exists":
        passed = True
    elif operator in {"lt", "lte", "gt", "gte"}:
        actual_num, expected_num = _number(actual), _number(expected)
        if actual_num is None or expected_num is None:
            return {"requirement": requirement, "attribute": attribute_name,
                    "status": "unknown", "actual": actual}
        passed = {"lt": actual_num < expected_num, "lte": actual_num <= expected_num,
                  "gt": actual_num > expected_num, "gte": actual_num >= expected_num}[operator]
    elif matched_expected and operator in {"eq", "contains"}:
        passed = True
    else:
        actual_text, expected_text = _s(actual).lower(), _s(expected).lower()
        if not expected_text:
            return {"requirement": requirement, "attribute": attribute_name,
                    "status": "unknown", "actual": actual}
        passed = expected_text in actual_text if operator == "contains" else actual_text == expected_text
    return {"requirement": requirement, "attribute": resolved_attribute,
            "status": "pass" if passed else "fail", "actual": actual}


def evaluate_hard_constraints(intent: dict, hit: dict) -> list[dict]:
    hard = [item for item in (intent.get("constraints") or [])
            if item.get("strength") == "hard"]
    return [_constraint_check(item, hit.get("attributes") or {}) for item in hard]


def _same_requirement(left: str, right: str) -> bool:
    stopwords = {"must", "have", "need", "they", "with", "that", "specific",
                 "offer", "shoes", "running", "under", "than", "from"}
    tokens = lambda text: {token for token in re.findall(r"[a-z0-9]+", _s(text).lower())
                           if len(token) > 2 and token not in stopwords}
    return bool(tokens(left) & tokens(right))

RERANK_SYSTEM_PROMPT = """You are the recommendation step of an AI shopping assistant.
A shopper request has already been decomposed into an authoritative intent structure. You are
given candidate products and, for each, the ONLY content available about it.

Decide which you would actually recommend, in order.

Hard rules:
- Justify a pick ONLY with information actually present in that candidate's content.
  Never infer, estimate, or invent a spec, number, material or claim.
- Judge purely on the content given. Ignore anything you may know about these brands
  or products from elsewhere — if the content does not say it, it is not available.
- Treat the supplied intent structure as immutable. Preserve its task action and framing in every
  explanation. In particular, never change running/participating in an event into training for it.
- Evaluate task compatibility first, context second, hard constraints third, then preferences.
- A derived suitability judgment may combine supplied facts, but describe the reasoning; never
  falsely say a product was designed for a task when the content only supports an inference.
- Treat a requirement contradicted by known content (for example, price 219 when the budget
  ceiling is 200) as a failed requirement, not an unanswered requirement.
- Use "unanswered" only when the content lacks the information needed to evaluate a shopper
  requirement. Never put the same requirement in both lists.
- Set "recommend" to true only when every explicit hard requirement is supported and none fail.
- Verified hard-constraint checks supplied with each candidate are authoritative. Never contradict
  a pass, invent a target value, or mark an `exists` constraint failed because values differ.
- Rank supported recommendations before products that fail or cannot answer hard requirements.
- Interpret the complete request, not isolated keywords. A gym, indoor, recovery, trail, or casual
  product is not proven suitable for half-marathon training merely because it is cheap and light.
- For distance or activity tasks, require compatible use-case/surface evidence. For climate
  requests, require relevant evidence such as ventilation or breathability.
- Do not call a product "lightweight" just because a weight is present. Compare known weights in
  this candidate set and prefer the lighter products that also satisfy the use case, climate, and
  budget. Never sacrifice a hard use-case requirement merely to choose the lowest weight or price.

Return exactly the best {result_limit} candidates (or all candidates if fewer were supplied).
Use candidate_id exactly as supplied; do not return product names or keys in its place.
Return strict JSON:
{"ranked": [
  {"candidate_id": 0,
   "recommend": true or false,
   "reason": "one sentence citing the specific attributes and values that justified this",
   "cited_attributes": ["attribute name or value quoted from the content", ...],
   "failed_requirements": ["requirement contradicted by a known product value", ...],
   "unanswered": ["what the shopper asked about that this content does not address", ...]}
]}
Rank the selected candidates best first."""


def rerank(query: str, intent: dict, hits: list[dict], limit: int = TOP_K) -> list[dict]:
    """Ask the model to order the retrieved candidates and cite what justified each pick.

    Retrieval alone yields cosine scores, which say nothing about WHY a product matched.
    This step is what turns the demo from 'here are some vectors' into 'here is the
    attribute that won it, and here is what the content still cannot answer'."""
    if not hits:
        return []

    evaluated = []
    for hit in hits:
        candidate = dict(hit)
        candidate["constraint_checks"] = evaluate_hard_constraints(intent, candidate)
        evaluated.append(candidate)
    eligible = [hit for hit in evaluated
                if all(check["status"] == "pass" for check in hit["constraint_checks"])]
    # Hard constraints are a gate. Generated prose and model judgment cannot override it.
    hits = eligible if eligible else evaluated

    candidates = "\n\n".join(
        f"[candidate_id: {candidate_id}]\nName: {h['name']}\n"
        f"Verified hard-constraint checks: {json.dumps(h.get('constraint_checks', []), ensure_ascii=False)}\n"
        f"Content on file:\n{h['text'][:2400]}"
        for candidate_id, h in enumerate(hits)
    )
    result = call_llm_json(
        RERANK_SYSTEM_PROMPT.replace("{result_limit}", str(min(limit, len(hits)))),
        f"Original shopper request:\n{query}\n\n"
        f"AUTHORITATIVE INTENT STRUCTURE:\n{json.dumps(intent, ensure_ascii=False)}\n\n"
        f"Return the best {min(limit, len(hits))} candidates.\n\nCandidate products:\n\n{candidates}",
        temperature=0.1,
    )
    ranked = result.get("ranked", []) if isinstance(result, dict) else []

    ordered = []
    seen = set()
    for item in ranked:
        candidate_id = item.get("candidate_id")
        try:
            candidate_id = int(candidate_id)
        except (TypeError, ValueError):
            continue
        if candidate_id < 0 or candidate_id >= len(hits) or candidate_id in seen:
            continue
        seen.add(candidate_id)
        hit = dict(hits[candidate_id])
        hit.update({
            "final_rank": len(ordered) + 1,
            "recommend": bool(item.get("recommend", False)),
            "reason": _s(item.get("reason")),
            "cited_attributes": [_s(a) for a in (item.get("cited_attributes") or [])],
            "failed_requirements": [_s(a) for a in (item.get("failed_requirements") or [])],
            "unanswered": [_s(a) for a in (item.get("unanswered") or [])],
        })
        passed = [check["requirement"] for check in hit.get("constraint_checks", [])
                  if check["status"] == "pass"]
        # The model may not turn a verified pass back into a failure or unknown.
        hit["failed_requirements"] = [item for item in hit["failed_requirements"]
                                      if not any(_same_requirement(item, req) for req in passed)]
        hit["unanswered"] = [item for item in hit["unanswered"]
                             if not any(_same_requirement(item, req) for req in passed)]
        failed = [check["requirement"] for check in hit.get("constraint_checks", [])
                  if check["status"] == "fail"]
        unknown = [check["requirement"] for check in hit.get("constraint_checks", [])
                   if check["status"] == "unknown"]
        if failed or unknown:
            hit["recommend"] = False
            hit["failed_requirements"] = list(dict.fromkeys(
                hit["failed_requirements"] + failed))
            hit["unanswered"] = list(dict.fromkeys(hit["unanswered"] + unknown))
        ordered.append(hit)
        if len(ordered) >= limit:
            break

    # A malformed response should remain visibly degraded, not masquerade as a valid ranking.
    if not ordered:
        for position, hit in enumerate(hits[:limit], 1):
            extra = dict(hit)
            extra.update({
                "final_rank": position,
                "recommend": False,
                "reason": "Assistant ranking response unavailable; showing semantic retrieval order.",
                "cited_attributes": [],
                "failed_requirements": [],
                "unanswered": [],
            })
            ordered.append(extra)
    return ordered


def run_query(raw_index: dict, optimized_index: dict, query: str, k: int = TOP_K) -> dict:
    """Embed the query once, search both indexes, rerank both. Returns the before/after pair
    plus the rank movement per product that appears in both."""
    query_vec = embed_query(query)
    available_attributes = {}
    for index in (raw_index, optimized_index):
        for entry in index.get("entries", []):
            for key, value in (entry.get("attributes") or {}).items():
                values = available_attributes.setdefault(key, [])
                value_text = _s(value).strip()
                if value_text and value_text not in values and len(values) < 8:
                    values.append(value_text)
    intent = parse_shopper_intent(query, available_attributes)

    candidate_k = max(k, CANDIDATE_K)
    raw_hits = rerank(query, intent, search(raw_index, query_vec, candidate_k), limit=k)
    optimized_hits = rerank(query, intent, search(optimized_index, query_vec, candidate_k), limit=k)

    raw_rank = {h["key"]: h["final_rank"] for h in raw_hits}
    movement = []
    for hit in optimized_hits:
        before = raw_rank.get(hit["key"])
        movement.append({
            "name": hit["name"],
            "before": before,          # None = did not surface at all on raw content
            "after": hit["final_rank"],
        })

    return {"query": query, "intent": intent, "raw": raw_hits,
            "optimized": optimized_hits, "movement": movement}
