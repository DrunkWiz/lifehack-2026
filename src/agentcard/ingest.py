"""Shopify-style CSV -> normalised catalogue rows.

Deliberately dumb. Ingest does not try to parse the spec metafield into typed
attributes; that heterogeneous string is exactly what the enricher is for. All
ingest guarantees is a stable id, a category, and clean text fields.
"""
from __future__ import annotations
import csv, html, json, pathlib, re
from .config import CATALOGUES, FIXTURES

_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(s: str) -> str:
    return html.unescape(_TAG_RE.sub(" ", s or "")).strip()


def load_reviews() -> dict[str, list[str]]:
    p = FIXTURES / "reviews.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def read_catalogue(csv_path: pathlib.Path, category: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("Handle"):
                continue
            rows.append({
                "id": r["Handle"],
                "sku": r.get("Variant SKU", ""),
                "category": category,
                "title": r.get("Title", ""),
                "brand": r.get("Vendor", ""),
                "product_type": r.get("Type", ""),
                "description": strip_html(r.get("Body (HTML)", "")),
                "tags": [t.strip() for t in (r.get("Tags") or "").split(",") if t.strip()],
                "price": float(r.get("Variant Price") or 0),
                "currency": "SGD",
                "inventory_qty": int(r.get("Variant Inventory Qty") or 0),
                "variant_grams": r.get("Variant Grams", ""),
                "option": f'{r.get("Option1 Name","")}: {r.get("Option1 Value","")}',
                "specs_text": r.get("Specs (product.metafields.custom.specs)", ""),
                "ingredients_text": r.get("Ingredients (product.metafields.custom.ingredients)", ""),
                "seo_description": r.get("SEO Description", ""),
            })
    return rows


def load_all() -> list[dict]:
    out = []
    for category, cfg in CATALOGUES.items():
        if cfg["csv"].exists():
            out += read_catalogue(cfg["csv"], category)
    return out


def raw_text(row: dict) -> str:
    """Everything an agent could scrape from the product page today."""
    return " \n".join(filter(None, [
        row["title"], row["brand"], row["product_type"], row["description"],
        ", ".join(row["tags"]), row["specs_text"], row["ingredients_text"],
        row["seo_description"], f'{row["currency"]} {row["price"]:.2f}',
    ]))
