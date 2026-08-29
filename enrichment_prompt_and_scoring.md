# Enrichment Prompt + Readiness Scoring

## 1. System prompt for the enricher

Use structured outputs with `agent_card_schema.json` enforced. Do not let the model free-form.

```
You convert raw retail catalog data into Agent Cards: structured representations
that let an AI shopping assistant reason about a product and decide whether to
recommend it for a specific shopper intent.

RULES

1. GROUNDING. Every hard_constraint must come from the input data. If a spec is
   absent, omit it. Do not estimate weights, dimensions, or ingredient
   percentages. An omitted field scores better than a wrong one.

2. PROVENANCE. Tag every field: catalog_spec (stated in input),
   review_derived (supported by supplied review text), inferred (reasoned from
   category knowledge). Anything inferred that a shopper could act on and be
   harmed by belongs in provenance.unsupported_claims.

3. NEGATIVE INFORMATION IS MANDATORY. Produce at least two not_for entries.
   Marketing copy never states who a product is wrong for; this is the field
   that prevents bad recommendations. Be specific: "runners with wide feet
   (E width or above)", not "some people".

4. USE CASES MUST CITE SPECS. Every why_it_fits references a hard_constraint
   key via grounded_in. "Great for humid weather" is worthless. "Engineered mesh
   upper with 4mm perforations, tested at 80%+ humidity" is usable.

5. SITUATIONAL TAGS bridge lifestyle language to specs. A shopper says "humid
   Singapore weather"; the catalog says "engineered mesh". Emit
   situational_tags: ["humid_climate", "tropical_training"] so retrieval can
   connect them. Use snake_case, reusable across the catalog.

6. COMPARISONS need a tradeoff. "Lighter than X" is incomplete. "38g lighter
   than X, with less heel cushioning above 15km" is a reasoning aid.

7. PERSONAS carry fit ratings including "poor". A card where every persona is
   a strong fit is a card that helps no one rank anything.

OUTPUT: valid JSON conforming to the supplied schema. No prose, no markdown.
```

## 2. User message shape

```
CATEGORY CONFIG:
{category_config_yaml}

RAW CATALOG ROW:
{csv_row_as_json}

REVIEW EXCERPTS (may be empty):
{reviews}

COMPETITOR CONTEXT (siblings in same price band):
{competitor_titles_and_key_specs}
```

## 3. Category config pattern (this is your scalability answer)

Core schema never changes. Only this file does. Show a judge two of these
side by side and rubric 4 is settled.

```yaml
# configs/footwear_running.yaml
category: footwear.running
required_numeric: [weight_g, heel_drop_mm, stack_height_mm]
required_categorical: [surface, arch_support, width, closure]
situational_vocabulary:
  - humid_climate
  - tropical_training
  - road_long_distance
  - race_day
  - recovery_run
persona_axes: [experience_level, weekly_mileage, foot_shape, budget_sensitivity]
common_exclusions: [wide_feet, overpronation, trail_use, heavy_runner]
```

```yaml
# configs/skincare_facial.yaml
category: skincare.facial
required_numeric: [volume_ml, routine_time_min, ph]
required_categorical: [skin_type, key_actives, fragrance_free, comedogenic_rating]
situational_vocabulary:
  - oily_skin_humid_climate
  - morning_routine_under_5min
  - sensitive_barrier_repair
  - layerable_under_spf
persona_axes: [skin_type, sensitivity, routine_complexity_tolerance, sustainability_priority]
common_exclusions: [pregnancy, active_retinoid_use, fragrance_sensitivity, dry_skin]
```

Adding a category = writing 15 lines of YAML. Say that sentence at the booth.

## 4. Readiness Score

Score 0-100, five components. Compute deterministically in Python, not with an
LLM. Judges will ask if the score is just another model guessing; "it's a
formula over the card" is the answer you want.

```python
WEIGHTS = {
    "attribute_completeness": 0.25,   # required_* fields present per category config
    "use_case_coverage":      0.25,   # distinct scenarios, capped at 6
    "persona_coverage":       0.15,   # distinct personas incl. >=1 poor fit
    "comparative_context":    0.15,   # comparisons with tradeoffs, capped at 3
    "claim_grounding":        0.20,   # share of fields with catalog_spec/review_derived
}

def readiness(card, config):
    req = config["required_numeric"] + config["required_categorical"]
    present = sum(1 for k in req if has_constraint(card, k))
    attr = present / len(req)

    uc   = min(len(card["use_cases"]), 6) / 6
    pers = min(len(card["personas"]), 4) / 4
    if not any(p["fit"] == "poor" for p in card["personas"]):
        pers *= 0.7                       # penalise all-positive cards

    comp = min(sum(1 for c in card["comparisons"] if c.get("tradeoff")), 3) / 3

    srcs = [f["source"] for f in card["provenance"]["field_sources"]]
    grounded = sum(1 for s in srcs if s in ("catalog_spec", "review_derived"))
    ground = grounded / max(len(srcs), 1)
    ground *= (1 - 0.1 * len(card["provenance"]["unsupported_claims"]))

    parts = {"attribute_completeness": attr, "use_case_coverage": uc,
             "persona_coverage": pers, "comparative_context": comp,
             "claim_grounding": max(ground, 0)}
    return round(100 * sum(WEIGHTS[k] * v for k, v in parts.items()), 1), parts
```

Run it on the RAW row too, by building a degenerate card from the CSV alone.
That gives you the before/after pair that sells the whole product.

## 5. Simulator harness

```
for each persona x situation in category config:
    generate 20 natural-language queries (cheap model, high temperature)
run retrieval twice: raw corpus vs enriched corpus
metric: recall@3 for the ground-truth product
report: raw X% -> enriched Y%
```

Ground truth comes from generating each query *from* a known product, then
checking whether retrieval finds its way back. Cheap, defensible, and it
produces the number for your poster.

## 6. Cost control

- Cache every completion to disk keyed by `sha256(prompt)`. Non-negotiable.
- Set a $30 hard limit in the OpenAI dashboard before writing any code.
- Strong model for enrichment (~300 cards). Cheapest model for query
  generation and relevance judging (~600 calls). Embeddings on
  text-embedding-3-small.
- Realistic total: under $5.
