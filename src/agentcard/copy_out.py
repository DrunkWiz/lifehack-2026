"""Turn an Agent Card into things a brand can actually paste somewhere.

Three surfaces, all rendered from the card with no further model call:

  passage   the semantic paragraph an assistant can quote, plus "Best for" and
            a Q&A pair — the format the sibling prototype generated, which is
            the right format for a marketer
  json_ld   schema.org Product markup for the product page itself, so answer
            engines crawling the page read structure rather than prose
  review    what a human must check before publishing

The distinction from generating this with a model: every sentence here is
assembled from fields that already exist in the card, so nothing new can be
invented at this step, and anything the enricher marked `inferred` is labelled
rather than asserted. The prototype's generator produced "dermatologist tested
and fragrance-free, making it suitable for sensitive skin types" as flat fact.
The first half is a spec, the second is an inference about skin safety, and a
brand's compliance team needs to see which is which.
"""
from __future__ import annotations
import json, re
from . import schema

_UNIT_SUFFIX = re.compile(r"_(g|kg|ml|l|mm|cm|min|pct|spf)$")

_SOURCE_LABEL = {"catalog_spec": "stated in the catalogue",
                 "review_derived": "from customer reviews",
                 "inferred": "inferred — verify before publishing",
                 "brand_asset": "from brand assets"}


def _sources(card: dict) -> dict[str, str]:
    return {f.get("field_path", ""): f.get("source", "")
            for f in card.get("provenance", {}).get("field_sources", [])}


def _constraint_sentence(card: dict) -> str:
    bits = []
    for n in card["hard_constraints"].get("numeric", [])[:4]:
        key, unit = n["key"], (n.get("unit") or "")
        # "volume_ml 50ml" and "ph 5.2pH" both read badly: the unit is already
        # in the key, or is the key.
        label = _UNIT_SUFFIX.sub("", key).replace("_", " ")
        if unit.lower() == key.lower():
            unit = ""
        bits.append(f'{label} {n["value"]}{unit}')
    for c in card["hard_constraints"].get("categorical", [])[:3]:
        vals = ", ".join(map(str, c.get("values", [])))
        if vals:
            bits.append(f'{c["key"].replace("_", " ")} {vals}')
    return "; ".join(bits)


def passage(card: dict, intent: str | None = None) -> str:
    """The paste-into-your-PDP block."""
    idy = card["identity"]
    uses = card.get("use_cases") or []
    chosen = uses[0] if uses else None
    if intent:
        chosen = next((u for u in uses if intent.lower() in u.get("scenario", "").lower()),
                      chosen)

    lines = []
    pitch = card.get("narrative", {}).get("one_line_pitch", "")
    specs = _constraint_sentence(card)
    why = chosen["why_it_fits"] if chosen else ""
    # Only restate the specs when the grounded sentence has not already done it,
    # otherwise the paragraph says the same numbers twice.
    restate = specs and not any(ch.isdigit() for ch in why)
    body = " ".join(x for x in [
        pitch, why, f"Specifications: {specs}." if restate else "",
    ] if x)
    lines.append(body.strip())

    if chosen:
        lines.append(f'\nBest for: {chosen["scenario"]}.')

    # The Q&A comes from not_for, because the question a shopper actually has is
    # "is this wrong for me?", and it is the question marketing copy never answers.
    nf = card.get("not_for") or []
    if nf:
        x = nf[0]
        # Phrased so it stays grammatical whatever shape the exclusion takes —
        # "Is this right for oily-skinned shopper in a tropical city?" does not.
        lines.append(f'\nQ: Who should skip this?'
                     f'\nA: {x["exclusion"]} — {x["reason"]}.')
    if len(nf) > 1:
        y = nf[1]
        lines.append(f'\nQ: Any other reason to look elsewhere?'
                     f'\nA: {y["exclusion"]} — {y["reason"]}.')

    unsupported = card.get("provenance", {}).get("unsupported_claims") or []
    if unsupported:
        lines.append("\n[Review before publishing: " + "; ".join(unsupported) + "]")
    return "\n".join(lines).strip()


def json_ld(card: dict) -> dict:
    """schema.org Product markup.

    This is the concrete AEO artefact: a brand pastes it into the product page
    and an answer engine crawling that page reads typed values instead of
    guessing at prose. additionalProperty carries the constraints that have no
    schema.org equivalent, which is most of what makes a product recommendable.
    """
    idy = card["identity"]
    price = idy.get("price") or {}
    props = []
    for n in card["hard_constraints"].get("numeric", []):
        props.append({"@type": "PropertyValue", "name": n["key"],
                      "value": n.get("value"), "unitText": n.get("unit", "")})
    for c in card["hard_constraints"].get("categorical", []):
        props.append({"@type": "PropertyValue", "name": c["key"],
                      "value": ", ".join(map(str, c.get("values", [])))})
    for t in schema.situational_tags(card):
        props.append({"@type": "PropertyValue", "name": "situational_tag", "value": t})

    availability = {"in_stock": "https://schema.org/InStock",
                    "low_stock": "https://schema.org/LimitedAvailability",
                    "out_of_stock": "https://schema.org/OutOfStock"}.get(
                        idy.get("availability", ""), "https://schema.org/InStock")

    doc = {
        "@context": "https://schema.org", "@type": "Product",
        "name": idy.get("title"), "sku": card.get("id"),
        "brand": {"@type": "Brand", "name": idy.get("brand")},
        "category": idy.get("category"),
        "description": card.get("narrative", {}).get("one_line_pitch", ""),
        "additionalProperty": props,
        "offers": {"@type": "Offer", "price": price.get("amount"),
                   "priceCurrency": price.get("currency", "SGD"),
                   "availability": availability},
    }
    faq = []
    for n in (card.get("not_for") or [])[:3]:
        faq.append({"@type": "Question",
                    "name": f'Is this suitable for {n["exclusion"].lower().rstrip(".")}?',
                    "acceptedAnswer": {"@type": "Answer", "text": f'No. {n["reason"]}.'}})
    for u in (card.get("use_cases") or [])[:3]:
        faq.append({"@type": "Question", "name": f'Is this good for {u["scenario"].lower()}?',
                    "acceptedAnswer": {"@type": "Answer", "text": u["why_it_fits"]}})
    if faq:
        doc["subjectOf"] = {"@type": "FAQPage", "mainEntity": faq}
    return doc


def review_notes(card: dict) -> list[str]:
    """What a human should check. Inferred claims first — they are the risk."""
    out = []
    for u in card.get("provenance", {}).get("unsupported_claims", []) or []:
        out.append(f"UNSUPPORTED — {u}")
    for f in card.get("provenance", {}).get("field_sources", []):
        if f.get("source") == "inferred":
            out.append(f'inferred — {f.get("field_path")} '
                       f'({f.get("evidence") or "no evidence recorded"})')
    for n in card.get("not_for", []):
        if n.get("source") == "inferred":
            out.append(f'inferred exclusion — "{n["exclusion"]}": {n["reason"]}')
    return out


def bundle(card: dict, intent: str | None = None) -> dict:
    return {"passage": passage(card, intent),
            "json_ld": json_ld(card),
            "review": review_notes(card),
            "provenance_summary": _summarise_sources(card)}


def _summarise_sources(card: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for f in card.get("provenance", {}).get("field_sources", []):
        counts[f.get("source", "unknown")] = counts.get(f.get("source", "unknown"), 0) + 1
    return counts
