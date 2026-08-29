"""Catalogue -> Agent Cards."""
from __future__ import annotations
import json, sys
from . import config, ingest, llm, prompts, readiness, schema


def competitors_for(row: dict, rows: list[dict], n: int = 3) -> list[dict]:
    same = [r for r in rows if r["category"] == row["category"] and r["id"] != row["id"]]
    same.sort(key=lambda r: abs(r["price"] - row["price"]))
    return same[:n]


def enrich_row(row: dict, rows: list[dict], cat_cfg: dict, reviews: dict,
               provider: str | None = None, model: str | None = None) -> dict:
    user = prompts.build_user_message(row, cat_cfg, reviews.get(row["id"], []),
                                      competitors_for(row, rows))
    card = llm.complete_json(prompts.SYSTEM, user,
                             schema_format=schema.response_format(),
                             model=model, provider=provider)
    return readiness.attach(postprocess(card), cat_cfg)


def postprocess(card: dict) -> dict:
    """Undo the strict-mode workarounds.

    OpenAI structured outputs cannot express an open-ended object, so
    schema.to_strict() turns category_extension into a JSON string. Parse it
    back here so what lands on disk validates against the authored schema.
    """
    ce = card.get("category_extension")
    if isinstance(ce, str):
        try:
            card["category_extension"] = json.loads(ce) if ce.strip() else {}
        except json.JSONDecodeError:
            card["category_extension"] = {"_unparsed": ce}
    card.setdefault("provenance", {}).setdefault("field_sources", [])
    card["provenance"].setdefault("unsupported_claims", [])
    for key in ("use_cases", "personas", "not_for", "comparisons"):
        card.setdefault(key, [])
    return card


def run(provider: str | None = None, model: str | None = None,
        limit: int | None = None, verbose: bool = True) -> dict:
    rows = ingest.load_all()
    reviews = ingest.load_reviews()
    configs = config.all_category_configs()
    cards, baselines, errors = {}, {}, []

    todo = rows[:limit] if limit else rows
    for i, row in enumerate(todo, 1):
        cfg = configs[row["category"]]
        baselines[row["id"]] = readiness.attach(readiness.raw_baseline_card(row, cfg), cfg)
        try:
            card = enrich_row(row, rows, cfg, reviews, provider, model)
        except llm.BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001
            errors.append({"id": row["id"], "error": f"{type(e).__name__}: {e}"})
            if verbose:
                print(f"  [{i}/{len(todo)}] {row['id']}: FAILED — {e}", file=sys.stderr)
            continue
        problems = schema.validate(card)
        if problems:
            errors.append({"id": row["id"], "error": "schema: " + "; ".join(problems[:3])})
        cards[row["id"]] = card
        if verbose:
            print(f"  [{i}/{len(todo)}] {row['id']}: readiness "
                  f"{baselines[row['id']]['readiness']['score']:.0f} -> "
                  f"{card['readiness']['score']:.0f}"
                  + ("  (schema warnings)" if problems else ""))

    config.OUT.mkdir(parents=True, exist_ok=True)
    (config.OUT / "agent_cards.json").write_text(
        json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.OUT / "raw_baseline_cards.json").write_text(
        json.dumps(baselines, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "products": len(todo), "cards": len(cards), "errors": errors,
        "provider": provider or config.LLM_PROVIDER,
        "mean_readiness_raw": round(sum(b["readiness"]["score"] for b in baselines.values())
                                    / max(len(baselines), 1), 1),
        "mean_readiness_enriched": round(sum(c["readiness"]["score"] for c in cards.values())
                                         / max(len(cards), 1), 1),
        "spend_usd": round(llm.SPEND.usd, 4), "llm_calls": llm.SPEND.calls,
        "cache_hits": llm.SPEND.cached,
    }
    (config.OUT / "enrichment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary
