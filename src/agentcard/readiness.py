"""Content Readiness Score.

Deterministic. A formula over the card, not a model grading a model — that
distinction is the answer to the first question a judge asks.

Weights and shape follow enrichment_prompt_and_scoring.md §4.
"""
from __future__ import annotations
from .schema import has_constraint

WEIGHTS = {
    "attribute_completeness": 0.25,   # required_* fields present per category config
    "use_case_coverage":      0.25,   # distinct scenarios, capped at 6
    "persona_coverage":       0.15,   # distinct personas incl. >=1 poor fit
    "comparative_context":    0.15,   # comparisons with tradeoffs, capped at 3
    "claim_grounding":        0.20,   # share of fields with catalog_spec/review_derived
}

LABELS = {
    "attribute_completeness": "Filterable attributes",
    "use_case_coverage": "Use-case coverage",
    "persona_coverage": "Persona coverage",
    "comparative_context": "Comparative context",
    "claim_grounding": "Claim grounding",
}


def readiness(card: dict, config: dict) -> tuple[float, dict, list[str]]:
    req = list(config["required_numeric"]) + list(config["required_categorical"])
    missing = [k for k in req if not has_constraint(card, k)]
    attr = (len(req) - len(missing)) / max(len(req), 1)

    use_cases = card.get("use_cases") or []
    scenarios = {u.get("scenario", "").strip().lower() for u in use_cases if u.get("scenario")}
    uc = min(len(scenarios), 6) / 6

    personas = card.get("personas") or []
    labels = {p.get("label", "").strip().lower() for p in personas if p.get("label")}
    pers = min(len(labels), 4) / 4
    all_positive = personas and not any(p.get("fit") == "poor" for p in personas)
    if all_positive or not personas:
        pers *= 0.7                                  # penalise all-positive cards

    comps = card.get("comparisons") or []
    comp = min(sum(1 for c in comps if c.get("tradeoff")), 3) / 3

    srcs = [f.get("source") for f in card.get("provenance", {}).get("field_sources", [])]
    grounded = sum(1 for s in srcs if s in ("catalog_spec", "review_derived"))
    ground = grounded / max(len(srcs), 1)
    unsupported = card.get("provenance", {}).get("unsupported_claims", []) or []
    ground *= (1 - 0.1 * len(unsupported))

    parts = {"attribute_completeness": round(attr, 4), "use_case_coverage": round(uc, 4),
             "persona_coverage": round(pers, 4), "comparative_context": round(comp, 4),
             "claim_grounding": round(max(ground, 0), 4)}
    score = round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()), 1)

    gaps = []
    if missing:
        gaps.append("Missing filterable attributes: " + ", ".join(missing))
    if len(scenarios) < 4:
        gaps.append(f"Only {len(scenarios)} distinct use cases — agents need 4+ to match varied intent")
    if all_positive:
        gaps.append("No 'poor fit' persona — a card that suits everyone ranks for no one")
    if not any(c.get("tradeoff") for c in comps):
        gaps.append("No comparison states a tradeoff, so agents cannot rank this against siblings")
    if not card.get("not_for"):
        gaps.append("No negative information — nothing stops a wrong recommendation")
    if unsupported:
        gaps.append(f"{len(unsupported)} claim(s) flagged unsupported and pending review")
    gaps.sort(key=lambda g: -len(g))
    return score, parts, gaps[:5]


def attach(card: dict, config: dict) -> dict:
    score, parts, gaps = readiness(card, config)
    card["readiness"] = {"score": score, "components": parts, "top_gaps": gaps}
    return card


def raw_baseline_card(row: dict, config: dict) -> dict:
    """The 'before' card: everything a machine can extract from the catalogue
    row with no reasoning. This is what most brands are shipping today, and it
    is the honest comparison point for the readiness delta."""
    numeric, categorical, sources = [], [], []
    if row.get("variant_grams") and config["category"] == "footwear.running":
        try:
            numeric.append({"key": "weight_g", "value": float(row["variant_grams"]), "unit": "g"})
            sources.append({"field_path": "hard_constraints.numeric.weight_g",
                            "source": "catalog_spec", "evidence": "Variant Grams column"})
        except ValueError:
            pass
    if row.get("variant_grams") and config["category"] == "skincare.facial":
        try:
            numeric.append({"key": "volume_ml", "value": float(row["variant_grams"]), "unit": "ml"})
            sources.append({"field_path": "hard_constraints.numeric.volume_ml",
                            "source": "catalog_spec", "evidence": "Variant Grams column"})
        except ValueError:
            pass
    if row.get("tags"):
        categorical.append({"key": "tags", "values": row["tags"]})
        sources.append({"field_path": "hard_constraints.categorical.tags",
                        "source": "catalog_spec", "evidence": "Tags column"})
    price = row.get("price", 0.0)
    return {
        "id": row.get("sku") or row["id"],
        "identity": {"title": row["title"], "brand": row["brand"],
                     "category": config["category"],
                     "price": {"amount": price, "currency": "SGD",
                               "band": "budget" if price < 160 else "mid" if price < 240 else "premium"},
                     "availability": "in_stock" if row.get("inventory_qty", 0) > 0 else "out_of_stock"},
        "hard_constraints": {"numeric": numeric, "categorical": categorical, "situational_tags": []},
        "use_cases": [], "personas": [], "not_for": [], "comparisons": [],
        "narrative": {"one_line_pitch": row.get("description", "")[:160], "intent_variants": []},
        "category_extension": "{}",
        "provenance": {"field_sources": sources, "unsupported_claims": [],
                       "enriched_at": "", "model": "raw-catalogue-baseline"},
    }
