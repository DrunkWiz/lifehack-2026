"""Discover a catalogue's categories instead of being told them.

Ported from the sibling prototype, which had the better idea here: batch the
products, ask a model to group them, then run a second pass that merges labels
produced independently across batches so "Running Shoe" and "Running Shoes"
collapse into one cluster.

Paired with `infer_config`, this is what lets the pipeline run on a catalogue
nobody has written a config for: cluster the rows, ask what each cluster SHOULD
have, derive the config, enrich, index, retrieve.
"""
from __future__ import annotations
import collections, json, re
from . import config, llm

BATCH_SIZE = 20

CLUSTER_SYSTEM = """You group products into meaningful category clusters by what the
product IS and who it is for — semantic similarity, not keyword overlap.

Return strict JSON: {"clusters": [{"cluster_name": "...", "product_indices": [0,2,5]}]}
Every 0-based index given must appear in exactly one cluster. Cluster names are
short and human-readable: "Running Shoes", "Facial Moisturisers".
Prefer a handful of substantial clusters over many tiny ones; a cluster of one
product cannot be reasoned about as a category."""

MERGE_SYSTEM = """These cluster labels were produced independently across several
batches of one catalogue, so the same category may appear under several names.
Merge labels that refer to the same category.

Return strict JSON: {"merged": [{"final_name": "...", "source_names": ["...", ...]}]}
Every input label must appear under exactly one final_name."""

EXPECTED_SYSTEM = """Given a product category and a sample of real products in it, list
the attributes a well-informed buyer — or an AI shopping assistant reasoning on their
behalf — would expect to know before recommending one.

Pick 6-10 genuinely decision-relevant attributes for this category, not generic filler.
Include attributes that SHOULD be present even if these particular products lack them:
the purpose is to measure the gap, not to describe the sample.

Use snake_case with the unit implied by the name where there is one: weight_g,
volume_ml, ph, spf, routine_time_min.

Return strict JSON: {"expected_attributes": ["...", ...]}"""

_JSON = {"type": "json_object"}


def _listing(rows: list[dict], offset: int = 0) -> str:
    return "\n".join(
        f'{i}: {r.get("title","")} | type: {r.get("product_type","")} '
        f'| tags: {", ".join(r.get("tags", [])[:6])} '
        f'| {(r.get("description") or "")[:110]}'
        for i, r in enumerate(rows))


def _cluster_local(rows: list[dict]) -> list[dict]:
    """Offline fallback: group by product type, then by dominant tag.

    Crude but deterministic, and it keeps the whole flow runnable with no
    network — which is how the rest of this pipeline works.
    """
    buckets: dict[str, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        key = (r.get("product_type") or "").strip().lower()
        if not key:
            tags = [t for t in r.get("tags", []) if not t.replace(".", "").isdigit()]
            key = tags[0].lower() if tags else "uncategorised"
        buckets[key].append(i)
    return [{"cluster_name": k.title() or "Uncategorised", "product_indices": v}
            for k, v in sorted(buckets.items(), key=lambda kv: -len(kv[1]))]


def cluster(rows: list[dict], provider: str | None = None,
            model: str | None = None) -> list[dict]:
    provider = provider or config.LLM_PROVIDER
    if provider == "local":
        out = _cluster_local(rows)
        config.log(f"clustered {len(rows)} products into {len(out)} groups (local)")
        return out

    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    raw: list[dict] = []
    for b, batch in enumerate(batches):
        config.log(f"clustering batch {b+1}/{len(batches)}", indent=2)
        res = llm.complete_json(
            CLUSTER_SYSTEM,
            f"Group these {len(batch)} products:\n\n{_listing(batch)}",
            schema_format=_JSON, provider=provider, model=model or config.CHEAP_MODEL)
        for c in (res.get("clusters") or []) if isinstance(res, dict) else []:
            c["product_indices"] = [i + b * BATCH_SIZE for i in c.get("product_indices", [])]
            raw.append(c)

    if len(batches) > 1 and raw:
        labels = [c["cluster_name"] for c in raw]
        config.log(f"merging {len(set(labels))} labels across batches", indent=2)
        res = llm.complete_json(MERGE_SYSTEM, f"Cluster labels:\n{labels}",
                                schema_format=_JSON, provider=provider,
                                model=model or config.CHEAP_MODEL)
        groups = (res.get("merged") or []) if isinstance(res, dict) else []
        if groups:
            merged = []
            for g in groups:
                idx: list[int] = []
                for c in raw:
                    if c["cluster_name"] in g.get("source_names", []):
                        idx += c["product_indices"]
                if idx:
                    merged.append({"cluster_name": g["final_name"],
                                   "product_indices": sorted(set(idx))})
            raw = merged

    # Nothing may be silently dropped: a product missing from every cluster is a
    # product that never gets enriched, and the loss would be invisible.
    seen = {i for c in raw for i in c["product_indices"]}
    orphans = [i for i in range(len(rows)) if i not in seen]
    if orphans:
        config.log(f"{len(orphans)} product(s) fell out of clustering — kept as 'Other'")
        raw.append({"cluster_name": "Other", "product_indices": orphans})
    config.log(f"clustered {len(rows)} products into {len(raw)} groups")
    return raw


def expected_attributes(cluster_name: str, sample: list[dict],
                        provider: str | None = None,
                        model: str | None = None) -> list[str]:
    """What this category SHOULD carry, whether or not these rows do."""
    provider = provider or config.LLM_PROVIDER
    if provider == "local":
        return []
    body = "\n".join(f'- {r.get("title","")}: {r.get("specs_text","")[:160]}'
                     for r in sample[:8])
    res = llm.complete_json(EXPECTED_SYSTEM, f"Category: {cluster_name}\n\nSample:\n{body}",
                            schema_format=_JSON, provider=provider,
                            model=model or config.CHEAP_MODEL)
    attrs = (res.get("expected_attributes") or []) if isinstance(res, dict) else []
    return [re.sub(r"[^a-z0-9]+", "_", str(a).lower()).strip("_") for a in attrs if a]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "cluster"
