"""
Export: the integration path out of the tool.

Copy-pasting one product at a time from a text box is not an adoption story. Three formats,
each for a different consumer:

  CSV     - merchandisers, and re-import into a catalog/PIM
  JSON    - developers wiring this into their own pipeline; carries the full knowledge layer
  JSON-LD - schema.org Product markup, dropped into product pages. This is the one that
            matters: it is the format AI shopping surfaces and crawlers already consume, so
            the verified attributes become machine-readable at the source rather than living
            in this app.
"""

import io
import json
import re

CURRENCY_SYMBOLS = {"S$": "SGD", "US$": "USD", "$": "USD", "£": "GBP", "€": "EUR",
                    "¥": "JPY", "RM": "MYR", "₹": "INR", "A$": "AUD"}


def _s(val, default: str = "") -> str:
    return default if val is None else str(val)


def detect_currency(cluster_data: dict) -> str | None:
    """Read the currency off the price strings if the brand's export carries a symbol."""
    for data in cluster_data.values():
        for product in data.get("members", []):
            price = _s(product.get("price"))
            for symbol, code in CURRENCY_SYMBOLS.items():
                if symbol in price:
                    return code
    return None


def _price_number(price) -> str | None:
    m = re.search(r"\d+(?:[.,]\d+)?", _s(price).replace(",", ""))
    return m.group() if m else None


def _rows(cluster_data: dict):
    for cluster_name, data in cluster_data.items():
        content_map = data.get("content") or {}
        persona = data.get("selected_persona") or {}
        for product in data.get("members", []):
            name = _s(product.get("name"), "Unnamed product")
            yield cluster_name, data, persona, product, name, content_map.get(name)


def to_csv(cluster_data: dict) -> str:
    import pandas as pd
    records = []
    for cluster_name, data, persona, product, name, generated in _rows(cluster_data):
        normalized = product.get("specs_normalized") or {}
        records.append({
            "cluster": cluster_name,
            "product": name,
            "price": _s(product.get("price")),
            "persona": _s(persona.get("title")),
            "user_story": _s(data.get("user_story")),
            "agent_optimized_content": _s(generated),
            "verified_attributes": "; ".join(f"{k}: {v}" for k, v in normalized.items()),
            "original_description": _s(product.get("description")),
        })
    buf = io.StringIO()
    pd.DataFrame(records).to_csv(buf, index=False)
    return buf.getvalue()


def to_json(cluster_data: dict) -> str:
    """Full structured payload, knowledge layer included."""
    out = []
    for cluster_name, data, persona, product, name, generated in _rows(cluster_data):
        out.append({
            "cluster": cluster_name,
            "name": name,
            "price": _s(product.get("price")) or None,
            "original_description": _s(product.get("description")),
            "verified_attributes": product.get("specs_normalized") or {},
            "raw_attributes": product.get("specs") or {},
            "target_persona": {
                "title": _s(persona.get("title")) or None,
                "narrative_seed": _s(persona.get("narrative_seed")) or None,
                "user_story": _s(data.get("user_story")) or None,
            },
            "agent_optimized_content": _s(generated) or None,
        })
    return json.dumps({"products": out}, indent=2, ensure_ascii=False)


def to_jsonld(cluster_data: dict, currency: str | None = None) -> str:
    """schema.org Product objects, ready to embed as <script type="application/ld+json">.

    Verified attributes become additionalProperty/PropertyValue entries - the standard way to
    expose attributes a crawler or shopping agent can read without parsing prose."""
    graph = []
    for cluster_name, data, persona, product, name, generated in _rows(cluster_data):
        normalized = product.get("specs_normalized") or {}
        node = {
            "@type": "Product",
            "name": name,
            "category": cluster_name,
            "description": _s(generated) or _s(product.get("description")),
        }
        price = _price_number(product.get("price"))
        if price:
            offer = {"@type": "Offer", "price": price, "availability": "https://schema.org/InStock"}
            if currency:
                offer["priceCurrency"] = currency
            node["offers"] = offer
        if normalized:
            node["additionalProperty"] = [
                {"@type": "PropertyValue", "name": k, "value": _s(v)}
                for k, v in normalized.items() if _s(v).strip()
            ]
        if persona.get("title"):
            node["audience"] = {"@type": "PeopleAudience", "audienceType": _s(persona["title"])}
        graph.append(node)
    return json.dumps({"@context": "https://schema.org", "@graph": graph},
                      indent=2, ensure_ascii=False)
