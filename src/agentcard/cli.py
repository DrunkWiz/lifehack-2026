"""agentcard — command line entry point.

  python -m agentcard doctor      check what this machine can actually reach
  python -m agentcard enrich      catalogue -> Agent Cards (+ raw baselines)
  python -m agentcard index       build both corpora and the constraint store
  python -m agentcard simulate    recall@3, raw vs enriched vs enriched+sql
  python -m agentcard ask "..."   one query against the live retriever
  python -m agentcard all         enrich, index, simulate
  python -m agentcard compare     run both catalogue variants, report both lifts
  python -m agentcard infer-config  derive a category config from an unseen CSV
  python -m agentcard validate-score does readiness predict what gets recommended?
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from . import (config, enrich, index, infer_config, ingest, llm, retrieve,
               simulate, validate_score)


def _providers(args) -> tuple[str, str]:
    """Resolve providers, falling back to local if the API is unreachable."""
    llm_p = args.provider or config.LLM_PROVIDER
    emb_p = args.embed_provider or config.EMBED_PROVIDER
    if "openai" in (llm_p, emb_p):
        ok, why = llm.probe_openai()
        if not ok:
            print(f"! openai unreachable ({why}) — falling back to the local provider",
                  file=sys.stderr)
            llm_p = emb_p = "local"
    return llm_p, emb_p


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="agentcard", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["openai", "local"])
    p.add_argument("--embed-provider", choices=["openai", "local"])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor")
    e = sub.add_parser("enrich"); e.add_argument("--limit", type=int)
    e.add_argument("--model")
    sub.add_parser("index")
    s = sub.add_parser("simulate")
    s.add_argument("--per-product", type=int, default=8)
    s.add_argument("-k", type=int, default=3)
    a = sub.add_parser("ask"); a.add_argument("query")
    a.add_argument("-k", type=int, default=3)
    a.add_argument("--corpus", choices=["raw", "enriched"], default="enriched")
    a.add_argument("--no-filter", action="store_true")
    al = sub.add_parser("all")
    al.add_argument("--per-product", type=int, default=8)
    al.add_argument("--limit", type=int)
    cp = sub.add_parser("compare")
    cp.add_argument("--per-product", type=int, default=8)
    vs = sub.add_parser("validate-score")
    vs.add_argument("--arm", default="enriched_sql",
                    choices=["raw", "enriched", "enriched_sql"])
    ic = sub.add_parser("infer-config")
    ic.add_argument("--csv", required=True, help="catalogue to read")
    ic.add_argument("--label", required=True, help='e.g. "Yoga mats"')
    ic.add_argument("--category", help="dotted id; derived from --label if omitted")
    ic.add_argument("--out", help="write YAML here instead of stdout")

    args = p.parse_args(argv)

    if args.cmd == "doctor":
        ok, why = llm.probe_openai()
        print(json.dumps({
            "openai_reachable": ok, "detail": why,
            "llm_provider": config.LLM_PROVIDER, "embed_provider": config.EMBED_PROVIDER,
            "enrich_model": config.ENRICH_MODEL, "embed_model": config.EMBED_MODEL,
            "llm_base_url": config.LLM_BASE_URL or "api.openai.com (default)",
            "embed_base_url": config.EMBED_BASE_URL or "(same as llm)",
            "budget_usd": config.BUDGET_USD, "spend_so_far_usd": round(llm.SPEND.usd, 4),
            "cached_completions": len(list((config.CACHE / "completions").glob("*.json"))),
            "models": llm.check_models() if ok else "(skipped — API unreachable)",
        }, indent=2))
        if ok:
            bad = [v["model"] for v in llm.check_models().values() if not v["available"]]
            if bad:
                print(f"! configured models not available on this key: {', '.join(bad)}\n"
                      f"  set AGENTCARD_ENRICH_MODEL / _CHEAP_MODEL / _EMBED_MODEL in .env",
                      file=sys.stderr)
        return 0

    llm_p, emb_p = _providers(args)

    if args.cmd in ("enrich", "all"):
        print(f"enriching ({llm_p})...")
        summary = enrich.run(provider=llm_p, model=getattr(args, "model", None),
                             limit=getattr(args, "limit", None))
        print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, indent=2))
        if summary["errors"]:
            print(f"! {len(summary['errors'])} product(s) failed", file=sys.stderr)

    if args.cmd in ("index", "all"):
        print(f"indexing ({emb_p})...")
        print(json.dumps(index.build(embed_provider=emb_p), indent=2))

    if args.cmd in ("simulate", "all"):
        print("simulating...")
        rep = simulate.run(per_product=getattr(args, "per_product", 3),
                           k=getattr(args, "k", 3), provider=llm_p, embed_provider=emb_p)
        print(json.dumps(rep, indent=2))

    if args.cmd == "validate-score":
        rep = validate_score.run(arm=args.arm)
        print(json.dumps(rep, indent=2))
        return 0

    if args.cmd == "infer-config":
        import pathlib
        cat = args.category or ("custom." + args.label.lower().replace(" ", "_"))
        rows = ingest.read_catalogue(pathlib.Path(args.csv), cat)
        if not rows:
            print(f"! no rows read from {args.csv}", file=sys.stderr)
            return 1
        config.log(f"inferring a config from {len(rows)} rows ({llm_p})")
        cfg = infer_config.infer(rows, args.label, cat, provider=llm_p)
        problems = infer_config.validate(cfg)
        text = infer_config.to_yaml(cfg)
        if args.out:
            pathlib.Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
        for pr in problems:
            print(f"! {pr}", file=sys.stderr)
        return 0

    if args.cmd == "compare":
        return compare(args, llm_p, emb_p)

    if args.cmd == "ask":
        res = retrieve.search(args.query, k=args.k, corpus=args.corpus,
                              use_filter=not args.no_filter, embed_provider=emb_p)
        print(f"filters: {res['filters'].describe() or ['(none)']}  "
              f"-> {res['filtered_to']} candidates")
        for i, h in enumerate(retrieve.hydrate(res["hits"]), 1):
            print(f"{i}. {h['title']}  SGD {h['price']:.0f}  "
                  f"[score {h['score']:.3f}, readiness {h['readiness']:.0f}]")
            print(f"   {h['card'].get('narrative',{}).get('one_line_pitch','')}")
    return 0


def compare(args, llm_p: str, emb_p: str) -> int:
    """Run the whole pipeline against both catalogue variants.

    The point is not to pick the flattering number. A brand that already
    publishes structured specs gets less from enrichment than one that
    publishes marketing copy and a price, and both are worth stating — the
    second is the more common case and the stronger argument.

    Each variant runs in its own process because the output paths are resolved
    at import time from AGENTCARD_VARIANT.
    """
    results = {}
    for variant in config.VARIANTS:
        print(f"\n=== {variant} " + "=" * (60 - len(variant)), file=sys.stderr)
        env = dict(os.environ,
                   AGENTCARD_VARIANT=variant,
                   PYTHONPATH=str(config.ROOT / "src"))
        cmd = [sys.executable, "-m", "agentcard", "--provider", llm_p,
               "--embed-provider", emb_p, "all",
               "--per-product", str(args.per_product)]
        proc = subprocess.run(cmd, env=env, cwd=str(config.ROOT))
        if proc.returncode != 0:
            print(f"! {variant} failed", file=sys.stderr)
            return proc.returncode
        out = config.ROOT / "out" if variant == "spec_rich" else config.ROOT / "out" / variant
        results[variant] = json.loads((out / "simulator_report.json").read_text(encoding="utf-8"))

    summary = {
        "variants": {
            v: {"recall_at_k": r["recall_at_k"],
                "by_category": r["by_category"],
                "lift_points": r["lift_points"],
                "mrr": r["mrr"]}
            for v, r in results.items()},
        "note": ("spec_rich gives every catalogue row a custom metafield with weight, "
                 "heel drop, stack height, pH and comedogenic rating. typical is a "
                 "plain Shopify export — title, marketing body, tags, price, size. "
                 "The gap between the two lifts is how much of a brand's AI "
                 "readiness is already in their catalogue versus how much has to "
                 "be created."),
        "embed_provider": emb_p, "llm_provider": llm_p,
        "spend_usd": round(llm.SPEND.usd, 4),
    }
    (config.ROOT / "out" / "variant_comparison.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 68, file=sys.stderr)
    print(f"{'catalogue variant':<20}{'raw':>10}{'enriched':>12}{'+ SQL':>10}{'lift':>10}")
    for v, r in results.items():
        k = r["recall_at_k"]
        print(f"{v:<20}{k['raw']:>9.1f}%{k['enriched']:>11.1f}%"
              f"{k['enriched+sql']:>9.1f}%{r['lift_points']:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
