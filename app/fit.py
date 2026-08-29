"""
Persona fit: separates two things the old single "persona rating" collapsed into one number.

  COVERAGE - can an agent even answer this shopper's questions about this product?
             A content problem. The brand fixes it by supplying data.
  FIT      - given what we know, does the product actually satisfy what they asked for?
             A merchandising problem. No amount of rewriting makes a 340g shoe lightweight.

A cluster can be 95% coverage and 10% fit. That is not a failure, it is a finding, and the
two numbers point at different teams. Blending them into one score - as the old
30% completeness + 70% persona rating did - produced a number that meant neither.

The model is used once per persona to turn its stated need into machine-checkable criteria
against the normalized schema. Every product is then evaluated in plain Python, so scoring is
deterministic, inspectable, and free of per-product model calls.
"""

import re

from llm_utils import call_llm_json

NUMERIC_OPS = {"lt", "lte", "gt", "gte"}
ALL_OPS = NUMERIC_OPS | {"eq", "neq", "contains", "in", "present"}
_OP_ALIASES = {"<": "lt", "<=": "lte", ">": "gt", ">=": "gte", "=": "eq", "==": "eq", "!=": "neq"}


CRITERIA_SYSTEM_PROMPT = """You turn a shopper persona into machine-checkable criteria against a
fixed product attribute schema.

Given the persona and the schema (with example values so you can see the format the data uses),
express what this shopper actually requires as 2-5 criteria.

Each criterion is {"attribute", "operator", "value", "rationale"}:
- "attribute" MUST be one of the schema attributes given. Never invent one.
- "operator" is one of: lt, lte, gt, gte (numeric), eq, neq, contains (text), in (value is a
  list of acceptable options), present (the attribute merely needs to exist).
- "value" must match the format the real data uses - if weights appear as "258g", a threshold
  of 250 is right, not "250 grams". For "in", give a list.
- "rationale" is one short clause saying why this shopper needs it.

Choose criteria this persona would genuinely fail a product over. Do not pad to 5.

Return strict JSON: {"criteria": [{"attribute": "...", "operator": "...", "value": ..., "rationale": "..."}]}"""


def _num(value):
    """First number in a value string: '258g' -> 258.0, '1/5' -> 1.0, 'high' -> None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else None


def _text(value) -> str:
    return str(value).strip().lower()


def generate_fit_criteria(persona: dict, schema: list[str], products: list[dict]) -> list[dict]:
    """One model call per persona. Returns [] on failure - callers fall back to coverage only."""
    if not schema:
        return []
    examples = {}
    for product in products:
        for attr, value in (product.get("specs_normalized") or {}).items():
            if attr in schema and attr not in examples and value is not None:
                examples[attr] = value
    try:
        result = call_llm_json(
            CRITERIA_SYSTEM_PROMPT,
            f"Persona: {persona.get('title','')} — {persona.get('narrative_seed','')}\n"
            f"Schema attributes: {schema}\n"
            f"Example values from the real data: {examples}",
            temperature=0.1,
        )
    except Exception:
        return []
    if not isinstance(result, dict):
        return []

    criteria = []
    for c in result.get("criteria", []) or []:
        attr = str(c.get("attribute", "")).strip()
        if attr not in schema:
            continue                                    # invented attribute, drop it
        op = str(c.get("operator", "")).strip().lower()
        op = _OP_ALIASES.get(op, op)
        if op not in ALL_OPS:
            continue
        criteria.append({
            "attribute": attr,
            "operator": op,
            "value": c.get("value"),
            "rationale": str(c.get("rationale", "")).strip(),
        })
    return criteria[:5]


def _check(criterion: dict, value):
    """True / False / None. None means 'cannot evaluate' - the data is missing, which is a
    coverage failure, not a fit failure. The two are never conflated."""
    if value is None or str(value).strip() == "":
        return None
    op, target = criterion["operator"], criterion["value"]

    if op == "present":
        return True
    if op in NUMERIC_OPS:
        left, right = _num(value), _num(target)
        if left is None or right is None:
            return None
        return {"lt": left < right, "lte": left <= right,
                "gt": left > right, "gte": left >= right}[op]
    if op == "in":
        options = target if isinstance(target, (list, tuple)) else [target]
        return _text(value) in {_text(o) for o in options}
    if op == "contains":
        return _text(target) in _text(value)
    if op == "eq":
        return _text(value) == _text(target)
    if op == "neq":
        return _text(value) != _text(target)
    return None


def evaluate_fit(criteria: list[dict], products: list[dict]) -> dict:
    """Deterministic. Returns cluster-level coverage/fit plus a per-product breakdown."""
    if not criteria or not products:
        return {"criteria": criteria, "coverage_pct": 0.0, "fit_pct": 0.0,
                "qualifying": 0, "total": len(products), "per_product": []}

    per_product = []
    for product in products:
        normalized = product.get("specs_normalized") or {}
        passed, failed, unknown = [], [], []
        for criterion in criteria:
            verdict = _check(criterion, normalized.get(criterion["attribute"]))
            label = f"{criterion['attribute']} {criterion['operator']} {criterion['value']}"
            if verdict is None:
                unknown.append(label)
            elif verdict:
                passed.append(label)
            else:
                failed.append(label)

        evaluable = len(passed) + len(failed)
        per_product.append({
            "name": str(product.get("name") or "Unnamed product"),
            "passed": passed, "failed": failed, "unknown": unknown,
            # coverage: how much of what this shopper asks about we can actually answer
            "coverage_pct": round(100 * evaluable / len(criteria), 1),
            # fit: of what we CAN answer, how much satisfies them
            "fit_pct": round(100 * len(passed) / evaluable, 1) if evaluable else 0.0,
            # qualifies only if every criterion is both known AND satisfied
            "qualifies": len(passed) == len(criteria),
        })

    coverage = round(sum(p["coverage_pct"] for p in per_product) / len(per_product), 1)
    qualifying = sum(1 for p in per_product if p["qualifies"])
    return {
        "criteria": criteria,
        "coverage_pct": coverage,
        # cluster fit is the share of the range this shopper could actually be sold
        "fit_pct": round(100 * qualifying / len(products), 1),
        "qualifying": qualifying,
        "total": len(products),
        "per_product": per_product,
    }
