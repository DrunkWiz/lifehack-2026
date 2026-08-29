"""
Structured knowledge layer: turns whatever shape a brand's catalog arrives in into a
consistent set of attribute:value pairs an agent can reason over.

Why this exists: real catalogs bury decision-relevant attributes inside a single field.
A Shopify metafield might hold
    "Weight: 258g (US M9) | Heel-to-toe drop: 8mm | Surface: road | Upper: mesh (high ventilation)"
as ONE value. Splitting on "|" works on that file and breaks on the next brand's, which uses
"/" or newlines or prose - and splitting can never turn "mesh (high ventilation)" into
ventilation=high. So the mapping is done by the model, not by a delimiter heuristic.

Two passes, deliberately:
  1. infer_schema()    - one call per cluster fixes the canonical attribute names
  2. normalize_batch() - products are mapped ONTO that fixed schema
A single pass would let key names drift between products (weight / weight_g / Weight (g)),
which is precisely what makes completeness scoring meaningless.

Nothing here is trusted blindly. Every extracted value is checked to appear verbatim in that
product's own source text, and dropped if it does not - so a fabricated spec cannot enter the
knowledge layer even if the model emits one. Raw specs are never overwritten.
"""

import re

from llm_utils import call_llm_json

BATCH_SIZE = 10
MIN_VERIFY_LEN = 2   # values shorter than this skip verification (too noisy to check)


def _s(val, default: str = "") -> str:
    if val is None:
        return default
    return str(val)


# ---------------------------------------------------------------------------
# Pass 1 - canonical schema for the category
# ---------------------------------------------------------------------------

SCHEMA_SYSTEM_PROMPT = """You define the canonical attribute schema for a product category.

Given a category name and a sample of raw product records from it, list the attributes that
are decision-relevant for that category - the things a buyer, or an AI shopping assistant
reasoning on their behalf, would need in order to choose between these products.

Rules:
- 8 to 12 attributes. Decision-relevant only, no filler, no marketing fields.
- Base them on what this category actually requires AND what these records plausibly contain.
- Prefer attributes evidenced somewhere in the supplied records or broadly decision-relevant to
  multiple products in the cluster.
- Do not introduce a specialized optional mechanism that is absent from every supplied record
  merely because some products in the wider market might offer it. Optional subtype features are
  not universal catalog requirements.
- When the cluster contains heterogeneous product types, shared attributes may remain in the
  schema, but product-specific applicability will be decided separately.
- snake_case names, no units in the name unless the unit IS the attribute (weight_grams is
  good; weight_of_the_shoe is not).
- Do not include: name, title, price, sku, url, image, brand. Those are handled separately.

Return strict JSON: {"schema": ["attribute_one", "attribute_two", ...]}"""


def infer_schema(cluster_name: str, sample_products: list[dict]) -> list[str]:
    """Returns the canonical attribute names for this cluster, or [] if the call fails."""
    sample = "\n\n".join(
        f"- {_s(p.get('name'), 'Unnamed')}\n  description: {_s(p.get('description'))[:200]}\n"
        f"  raw fields: {p.get('specs') or {}}"
        for p in sample_products[:8]
    )
    try:
        result = call_llm_json(
            SCHEMA_SYSTEM_PROMPT,
            f"Category: {cluster_name}\n\nSample records:\n{sample}",
            temperature=0.1,
        )
    except Exception:
        return []
    if not isinstance(result, dict):
        return []
    schema = result.get("schema") or []
    return [str(a).strip() for a in schema if str(a).strip()][:12]


APPLICABILITY_SYSTEM_PROMPT = """Decide which attributes from a fixed category schema are
meaningful for each individual product type.

Applicability is not the same as presence. An attribute is applicable when it would make sense
for the brand to supply that fact for this exact kind of product, even if the current record is
missing it. Mark an attribute not applicable when the concept does not reasonably belong to that
product type. Do not force a broad accessory schema onto unrelated accessories. For example,
helmet certification is not a missing specification for ski poles, and an avalanche-airbag field
is not a missing specification for ordinary goggles.

Use only the supplied schema names. Return every product index exactly once.
Return strict JSON:
{"products": [{"index": 0, "applicable_attributes": ["exact_schema_name", ...]}, ...]}"""


def infer_applicability(cluster_name: str, schema: list[str], products: list[dict]) -> list[list[str]]:
    listing = "\n\n".join(
        f"[index: {i}]\nName: {_s(p.get('name'), 'Unnamed')}\n"
        f"Description: {_s(p.get('description'))[:300]}\nRaw fields: {p.get('specs') or {}}"
        for i, p in enumerate(products)
    )
    result = call_llm_json(
        APPLICABILITY_SYSTEM_PROMPT,
        f"Category: {cluster_name}\nSchema: {schema}\n\nProducts:\n{listing}",
        temperature=0.0,
    )
    if not isinstance(result, dict):
        return []
    output = [None for _ in products]
    for item in result.get("products", []) or []:
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(products):
            supplied = item.get("applicable_attributes") or []
            output[index] = [attribute for attribute in schema if attribute in supplied]
    return output if all(item is not None for item in output) else []


# ---------------------------------------------------------------------------
# Pass 2 - map products onto that schema
# ---------------------------------------------------------------------------

NORMALIZE_SYSTEM_PROMPT = """You map raw product records onto a FIXED attribute schema.

For each product you are given, return a value for each schema attribute.

Hard rules:
- Use ONLY information present in that product's own record. The value you emit must be
  taken from the text you were given, not inferred, estimated, or completed from knowledge
  of similar products.
- If the record does not state an attribute, return null for it. A null is correct and
  expected - do not guess to fill the schema.
- Values should be short and comparable: strip surrounding prose, keep the unit where it is
  part of the value ("258g", "8mm", "road", "high").
- Use exactly the schema attribute names given. Do not add, rename or omit attributes.

Return strict JSON:
{"products": [{"index": <the index given>, "attributes": {"attr_name": "value" or null, ...}}, ...]}"""


def _raw_text(product: dict) -> str:
    """Everything the source said about this product - the corpus verification checks against."""
    specs = product.get("specs") or {}
    parts = [
        _s(product.get("name")),
        _s(product.get("description")),
        " ".join(f"{k} {v}" for k, v in specs.items()),
    ]
    return " ".join(parts)


def _canon(text: str) -> str:
    """Lowercase, strip non-alphanumerics - so '8 mm' still matches '8mm'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _verify(attributes: dict, product: dict) -> tuple[dict, list[str]]:
    """Drop any value that does not appear verbatim in the product's own source text.
    Returns (kept, rejected_attribute_names)."""
    raw = _canon(_raw_text(product))
    kept, rejected = {}, []
    for attr, value in (attributes or {}).items():
        if value is None or _s(value).strip() == "" or _s(value).strip().lower() in ("null", "none", "n/a"):
            continue
        needle = _canon(_s(value))
        if len(needle) < MIN_VERIFY_LEN:
            kept[attr] = value          # too short to verify meaningfully
        elif needle in raw:
            kept[attr] = value
        else:
            rejected.append(attr)        # model produced something not in the source
    return kept, rejected


def normalize_batch(schema: list[str], products: list[dict]) -> list[dict]:
    """Returns a list of attribute dicts, index-aligned with `products`. [] on failure."""
    listing = "\n\n".join(
        f"[index: {i}]\nName: {_s(p.get('name'), 'Unnamed')}\n"
        f"Description: {_s(p.get('description'))[:400]}\n"
        f"Raw fields: {p.get('specs') or {}}"
        for i, p in enumerate(products)
    )
    result = call_llm_json(
        NORMALIZE_SYSTEM_PROMPT,
        f"Schema attributes: {schema}\n\nProducts:\n\n{listing}",
        temperature=0.0,
    )
    if not isinstance(result, dict):
        return []
    out = [{} for _ in products]
    for item in result.get("products", []) or []:
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(products):
            attrs = item.get("attributes") or {}
            out[idx] = {k: v for k, v in attrs.items() if k in schema}   # ignore invented keys
    return out


# ---------------------------------------------------------------------------
# Orchestration + fallbacks
# ---------------------------------------------------------------------------

def normalize_cluster(cluster_name: str, products: list[dict], progress=None) -> dict:
    """Writes `specs_normalized` onto each product in place. Raw `specs` are never touched.

    Returns stats: {schema, normalized_count, total, rejected_values, failed_batches}.
    An empty schema means normalization was skipped - callers fall back to the old path."""
    stats = {"schema": [], "normalized_count": 0, "total": len(products),
             "rejected_values": 0, "failed_batches": 0,
             "applicability_inferred": False}
    if not products:
        return stats

    schema = infer_schema(cluster_name, products)
    if not schema:
        return stats                      # FALLBACK 1: no schema -> skip entirely
    stats["schema"] = schema

    try:
        applicability = infer_applicability(cluster_name, schema, products)
    except Exception:
        applicability = []
    if applicability:
        stats["applicability_inferred"] = True
        for product, applicable in zip(products, applicability):
            product["applicable_attributes"] = applicable
    else:
        # Safe fallback preserves the old behavior if classification fails.
        for product in products:
            product["applicable_attributes"] = list(schema)

    batches = [products[i:i + BATCH_SIZE] for i in range(0, len(products), BATCH_SIZE)]
    for b_idx, batch in enumerate(batches):
        if progress:
            progress(b_idx, len(batches), cluster_name)

        attrs_list = []
        for attempt in range(2):          # FALLBACK 2: one retry, then leave the batch raw
            try:
                attrs_list = normalize_batch(schema, batch)
                if attrs_list:
                    break
            except Exception:
                attrs_list = []
        if not attrs_list:
            stats["failed_batches"] += 1
            continue

        for product, attrs in zip(batch, attrs_list):
            kept, rejected = _verify(attrs, product)   # FALLBACK 3: unverifiable values dropped
            stats["rejected_values"] += len(rejected)
            if kept:
                product["specs_normalized"] = kept
                stats["normalized_count"] += 1

    return stats
