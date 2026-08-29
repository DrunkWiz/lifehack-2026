"""Derive a category config from an unfamiliar catalogue.

The configs in `configs/` were written by hand, which is fine for two known
categories and useless for a brand that uploads a catalogue nobody has seen.
This module reads a set of raw rows and produces the same YAML: which
attributes are required, what shopper language maps to which constraint value,
which constraints are allowed to eliminate a product, and the words that say a
question belongs to this category.

Two paths, same output shape:
  openai — one structured-output call, reading a sample of rows
  local  — column statistics over the whole set, no network

The local path is genuinely useful, not just a stub: most of a config is
recoverable from what the rows actually contain. The model path is better at
the part statistics cannot reach — the situational vocabulary, which is about
shopper lives rather than product fields.
"""
from __future__ import annotations
import collections, json, re
import yaml
from . import config, llm

_KV = re.compile(r"([A-Za-z][A-Za-z0-9 /_-]{2,30})\s*[:=]\s*([^|,;]{1,40})")
_NUM = re.compile(r"^\s*([\d.]+)\s*([a-zA-Z%]{0,6})\s*$")
# Deliberately excludes "size" and "price": they look like filler words but are
# real, filterable attributes in most catalogues.
_STOP = {"the", "and", "for", "with", "your", "you", "our", "this", "that", "from",
         "all", "new", "more", "shop", "buy", "product", "products", "available"}

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string",
                     "description": "Dotted taxonomy id, e.g. footwear.running"},
        "label": {"type": "string", "description": "Human-readable name"},
        "required_numeric": {"type": "array", "items": {"type": "string"},
                             "description": "snake_case keys with units implied, "
                                            "e.g. weight_g, volume_ml, ph"},
        "required_categorical": {"type": "array", "items": {"type": "string"}},
        "situational_vocabulary": {
            "type": "array", "items": {"type": "string"},
            "description": "snake_case tags bridging shopper lifestyle language to "
                           "specs, e.g. humid_climate, morning_routine_under_5min"},
        "persona_axes": {"type": "array", "items": {"type": "string"}},
        "common_exclusions": {"type": "array", "items": {"type": "string"}},
        "hard_filter_keys": {
            "type": "array", "items": {"type": "string"},
            "description": "Only keys where a mismatch makes the product WRONG, not "
                           "merely less preferred. Be conservative — over-filtering "
                           "eliminates correct answers."},
        "category_cues": {
            "type": "array", "items": {"type": "string"},
            "description": "Words a shopper uses that mean the question is about "
                           "this category at all"},
        "constraint_synonyms_json": {
            "type": "string",
            "description": "JSON object: {constraint_key: {canonical_value: "
                           "[shopper phrases]}}. Lowercase phrases."},
    },
}

SYSTEM = """\
You write category configuration for a product-catalogue enrichment pipeline.

Given a sample of raw catalogue rows from one product category, produce the
config that lets an AI shopping assistant filter and rank them.

RULES

1. required_numeric and required_categorical are the attributes an assistant
   MUST have to decide whether a product fits. Name them in snake_case with the
   unit implied by the key (weight_g, volume_ml, ph, routine_time_min). Include
   attributes that SHOULD be present even if these rows lack them — the point is
   to measure the gap, not to describe the sample.

2. situational_vocabulary is where shopper lives meet product specs. Not
   attributes: situations. "humid_climate", "morning_routine_under_5min",
   "first_time_buyer". Ten to fifteen, reusable across the category.

3. hard_filter_keys is the dangerous one. List ONLY constraints where a
   mismatch makes a product genuinely wrong for the shopper — the wrong surface,
   an allergen. A preference like arch support or skin type must NOT be here; it
   ranks, it does not eliminate. Two or three keys at most.

4. constraint_synonyms_json maps the words shoppers actually type onto
   canonical constraint values. Include phrasings nobody writes in a spec sheet.

5. category_cues are words that identify a question as belonging to this
   category at all, so a query about shoes is not answered with moisturiser.

OUTPUT: JSON only.\
"""


def _sample(rows: list[dict], n: int = 12) -> list[dict]:
    step = max(1, len(rows) // n)
    return rows[::step][:n]


def infer_local(rows: list[dict], label: str, category: str,
                expected: list[str] | None = None) -> dict:
    """Recover what column statistics can reach. No network."""
    numeric, categorical = collections.Counter(), collections.Counter()
    values: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    # Brand names are the loudest tokens in any catalogue and the least useful:
    # a shopper describing a situation does not say "Lumen Lab", and a vocabulary
    # full of vendors routes every query to whoever sells the most SKUs.
    brands = set()
    for r in rows:
        for w in re.findall(r"[a-z]{3,}", (r.get("brand") or "").lower()):
            brands.add(w)

    for r in rows:
        blob = " | ".join(str(r.get(k, "")) for k in
                          ("specs_text", "option", "product_type"))
        for key, val in _KV.findall(blob):
            k = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
            if not k or k in _STOP:
                continue
            m = _NUM.match(val)
            if m:
                unit = m.group(2).lower()
                numeric[f"{k}_{unit}" if unit and not k.endswith(unit) else k] += 1
            else:
                categorical[k] += 1
                values[k][val.strip().lower()] += 1
        for t in r.get("tags", []):
            values["tags"][t.strip().lower()] += 1

    n = max(len(rows), 1)
    req_num = [k for k, c in numeric.most_common(8) if c >= n * 0.3]
    req_cat = [k for k, c in categorical.most_common(8) if c >= n * 0.3]

    # Attributes a category expert says SHOULD exist, whether or not these rows
    # carry them. Column statistics can only find what is already present, and
    # the entire point of a readiness score is to measure what is absent.
    for a in expected or []:
        if a in req_num or a in req_cat:
            continue
        (req_num if re.search(r"_(g|kg|ml|l|mm|cm|m|min|hrs?|months?|years?|days?|pct|ph|spf|count|rating)$", a) or a == "ph"
         else req_cat).append(a)

    def _is_brand(text: str) -> bool:
        toks = set(re.findall(r"[a-z]{3,}", text.lower()))
        return bool(toks) and toks <= brands

    tag_vocab = [t.replace(" ", "_").replace("-", "_")
                 for t, c in values["tags"].most_common(40)
                 if c >= 2 and not t.replace(".", "").isdigit() and not _is_brand(t)][:15]

    words = collections.Counter()
    for r in rows:
        for w in re.findall(r"[a-z]{4,}", f'{r.get("title","")} {r.get("product_type","")}'.lower()):
            if w not in _STOP:
                words[w] += 1
    cues = [w for w, c in words.most_common(30)
            if c >= max(2, n * 0.15) and w not in brands][:15]

    syn = {k: {v: [v] for v, _ in values[k].most_common(6)} for k in req_cat if values[k]}

    return {
        "category": category, "label": label,
        "required_numeric": req_num or ["price"],
        "required_categorical": req_cat or ["product_type"],
        "situational_vocabulary": tag_vocab,
        "persona_axes": ["experience_level", "budget_sensitivity", "usage_frequency"],
        "common_exclusions": [],
        # Empty on purpose. A wrong hard filter silently eliminates the correct
        # answer, and column statistics cannot tell a disqualifying attribute
        # from a preference. Everything ranks until a human — or the model path —
        # says otherwise, which fails safe.
        "hard_filter_keys": [],
        "category_cues": cues,
        "constraint_synonyms": syn,
        "_inferred_by": "local-column-statistics",
    }


def infer(rows: list[dict], label: str, category: str,
          provider: str | None = None, model: str | None = None,
          expected: list[str] | None = None) -> dict:
    provider = provider or config.LLM_PROVIDER
    if provider == "local":
        return infer_local(rows, label, category, expected)

    payload = json.dumps({"label": label, "category": category,
                          "attributes_a_category_expert_expects": expected or [],
                          "sample_rows": _sample(rows)}, indent=2, ensure_ascii=False)
    fmt = {"type": "json_schema", "json_schema": {
        "name": "category_config", "strict": True,
        "schema": {**CONFIG_SCHEMA, "additionalProperties": False,
                   "required": list(CONFIG_SCHEMA["properties"])}}}
    out = llm.complete_json(SYSTEM, payload, schema_format=fmt,
                            model=model, provider=provider)
    try:
        syn = json.loads(out.pop("constraint_synonyms_json", "{}") or "{}")
    except json.JSONDecodeError:
        syn = {}
    out["constraint_synonyms"] = syn
    out["_inferred_by"] = model or config.ENRICH_MODEL
    return out


REQUIRED_KEYS = ("category", "label", "required_numeric", "required_categorical",
                 "situational_vocabulary", "persona_axes", "common_exclusions",
                 "hard_filter_keys", "category_cues", "constraint_synonyms")


def validate(cfg: dict) -> list[str]:
    problems = [f"missing key: {k}" for k in REQUIRED_KEYS if k not in cfg]
    hard = set(cfg.get("hard_filter_keys", []))
    known = set(cfg.get("required_categorical", [])) | set(cfg.get("constraint_synonyms", {}))
    if not hard:
        problems.append("note: no hard_filter_keys — every constraint ranks rather "
                        "than eliminates. Safe, but a shopper saying 'fragrance-free' "
                        "will not have fragranced products removed. Add the "
                        "disqualifying keys once you know them.")
    for k in hard - known:
        problems.append(f"hard_filter_keys names '{k}', which is not a known "
                        f"categorical constraint — it would filter everything out")
    if len(hard) > 4:
        problems.append(f"{len(hard)} hard filter keys is too many; over-filtering "
                        f"eliminates correct answers")
    if not cfg.get("category_cues"):
        problems.append("no category_cues — queries cannot be routed to this category")
    return problems


def to_yaml(cfg: dict) -> str:
    ordered = {k: cfg[k] for k in REQUIRED_KEYS if k in cfg}
    ordered["_inferred_by"] = cfg.get("_inferred_by", "unknown")
    return yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=100)
