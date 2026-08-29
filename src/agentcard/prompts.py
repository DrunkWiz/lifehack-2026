"""The enrichment prompt. Kept verbatim from enrichment_prompt_and_scoring.md
so the spec and the code cannot drift."""
from __future__ import annotations
import json
import yaml

SYSTEM = """\
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
Prices are in SGD. category_extension must be a JSON object encoded as a
string. Leave readiness null-valued; it is computed deterministically in code.\
"""

USER_TEMPLATE = """\
CATEGORY CONFIG:
{category_config_yaml}

RAW CATALOG ROW:
{row_json}

REVIEW EXCERPTS (may be empty):
{reviews}

COMPETITOR CONTEXT (siblings in same price band):
{competitors}\
"""


def build_user_message(row: dict, category_config: dict,
                       reviews: list[str], competitors: list[dict]) -> str:
    comp_lines = [
        f'- {c["title"]} (SGD {c["price"]:.0f}) — {c["specs_text"] or "no published specs"}'
        for c in competitors
    ] or ["- (none in this price band)"]
    return USER_TEMPLATE.format(
        category_config_yaml=yaml.safe_dump(category_config, sort_keys=False).strip(),
        row_json=json.dumps(row, indent=2, ensure_ascii=False),
        reviews="\n".join(f"- {r}" for r in reviews) or "(none supplied)",
        competitors="\n".join(comp_lines),
    )


QUERY_GEN_SYSTEM = """\
You write realistic shopper questions for testing an AI shopping assistant.

Given one product's attributes, write natural-language questions a shopper
would type into an AI assistant that SHOULD surface this product.

RULES
- Never name the brand or the product. The test is whether retrieval finds it
  from intent alone.
- Describe the situation, not the specification. "training for my first half
  marathon in Singapore" beats "8mm drop neutral trainer".
- Vary the shape: some with a budget, some with a constraint ("I have wide
  feet"), some pure lifestyle, some comparative.
- One question per line. No numbering, no quotes, no commentary.\
"""
