"""Load a catalogue from whatever file a brand actually has.

CSV, XLSX, JSON (flat or Shopify `products.json`), or PDF. Ported and extended
from the sibling prototype in this project; the PDF path with its vision
fallback is that prototype's idea and the best thing in either codebase.

The important change is column mapping. The original matched header names by
exact equality — `price` only from a column literally called "price" — so a
Shopify export, whose columns are "Variant Price" and "Body (HTML)", produced a
null price, an empty description, and a specs blob full of raw HTML. Downstream
that reads as a catalogue with no data, and the readiness score ends up
measuring the loader rather than the catalogue. Header matching here is
alias-driven and scored, and it reports what it mapped so a human can correct it.
"""
from __future__ import annotations
import base64, html, io, json, math, pathlib, re
from . import config

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Ordered by preference. First match wins, and a header equal to the alias beats
# a header that merely contains it.
ALIASES: dict[str, list[str]] = {
    "title":        ["title", "product name", "name", "product", "item name", "product title"],
    "price":        ["variant price", "price", "unit price", "rrp", "msrp", "retail price",
                     "selling price", "amount"],
    "description":  ["body (html)", "body html", "description", "product description",
                     "desc", "details", "long description", "body"],
    "brand":        ["vendor", "brand", "manufacturer", "supplier", "maker"],
    "sku":          ["variant sku", "sku", "item code", "product code", "barcode", "mpn"],
    "tags":         ["tags", "keywords", "labels", "categories"],
    "product_type": ["type", "product type", "product category", "category", "subcategory"],
    "inventory":    ["variant inventory qty", "inventory quantity", "inventory", "stock",
                     "qty", "quantity", "available"],
    "grams":        ["variant grams", "grams", "weight", "net weight", "size", "volume"],
    "image":        ["image src", "image", "image url", "photo", "picture"],
    "seo":          ["seo description", "meta description", "seo title"],
    "handle":       ["handle", "slug", "url key", "id", "product id"],
    "ingredients":  ["ingredients", "inci", "composition", "materials", "fabric"],
    "specs":        ["specs", "specifications", "attributes", "features", "details"],
    "option_name":  ["option1 name", "option name"],
    "option_value": ["option1 value", "option value", "variant title"],
}


def _norm(h: str) -> str:
    return _WS.sub(" ", re.sub(r"[^a-z0-9() ]+", " ", str(h).lower())).strip()


def map_columns(headers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Return (field -> original header, unmapped headers).

    Scored rather than exact: an exact alias match beats a header that contains
    the alias, which beats nothing. Each header is claimed at most once.
    """
    normed = {h: _norm(h) for h in headers}
    chosen: dict[str, str] = {}
    taken: set[str] = set()
    for field, aliases in ALIASES.items():
        best, best_score = None, 0
        for h in headers:
            if h in taken:
                continue
            n = normed[h]
            for rank, alias in enumerate(aliases):
                weight = len(aliases) - rank          # earlier alias, stronger claim
                score = 0
                if n == alias:
                    score = 100 + weight
                elif n.startswith(alias) or n.endswith(alias):
                    score = 60 + weight
                elif alias in n:
                    score = 40 + weight
                if score > best_score:
                    best, best_score = h, score
        if best is not None:
            chosen[field] = best
            taken.add(best)
    return chosen, [h for h in headers if h not in taken]


def strip_html(s) -> str:
    if s is None:
        return ""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", str(s)))).strip()


def _clean(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "null") else s


def _price(v) -> float:
    s = _clean(v)
    m = re.search(r"(\d+(?:[.,]\d+)?)", s.replace(",", ""))
    return float(m.group(1)) if m else 0.0


def _int(v) -> int:
    s = _clean(v)
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else 0


def rows_from_records(records: list[dict], category: str,
                      headers: list[str] | None = None) -> tuple[list[dict], dict]:
    """Normalise tabular records into the pipeline's row shape."""
    headers = headers or (list(records[0].keys()) if records else [])
    cmap, unmapped = map_columns(headers)
    rows = []
    for i, r in enumerate(records):
        def get(field, default=""):
            h = cmap.get(field)
            return _clean(r.get(h)) if h else default

        # Anything we could not map is still information: keep it as a readable
        # spec string rather than discarding it or dumping raw HTML.
        leftovers = " | ".join(f"{h}: {strip_html(r.get(h))}" for h in unmapped
                               if _clean(r.get(h)) and len(str(r.get(h))) < 400)
        specs = get("specs")
        specs_text = " | ".join(x for x in (specs, leftovers) if x)

        title = get("title") or f"Product {i + 1}"
        handle = get("handle") or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or f"row-{i}"
        tags = [t.strip() for t in re.split(r"[,;|]", get("tags")) if t.strip()]
        rows.append({
            "id": handle, "sku": get("sku"), "category": category,
            "title": title, "brand": get("brand"),
            "product_type": get("product_type"),
            "description": strip_html(r.get(cmap["description"])) if "description" in cmap else "",
            "tags": tags, "price": _price(get("price")), "currency": "SGD",
            "inventory_qty": _int(get("inventory")) if "inventory" in cmap else 1,
            "variant_grams": get("grams"),
            "option": f'{get("option_name")}: {get("option_value")}'.strip(": "),
            "specs_text": specs_text,
            "ingredients_text": get("ingredients"),
            "seo_description": get("seo"),
        })
    report = {"mapped": {k: v for k, v in cmap.items()}, "unmapped": unmapped,
              "rows": len(rows),
              "missing_price": sum(1 for r in rows if not r["price"]),
              "missing_description": sum(1 for r in rows if not r["description"])}
    return rows, report


def rows_from_extracted(products: list[dict], category: str) -> list[dict]:
    """Map the {name, price, description, specs{}} shape from PDF/vision extraction."""
    rows = []
    for i, p in enumerate(products):
        specs = p.get("specs") or {}
        title = _clean(p.get("name")) or f"Product {i + 1}"
        rows.append({
            "id": re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or f"page-{i}",
            "sku": "", "category": category, "title": title,
            "brand": _clean(specs.get("vendor") or specs.get("brand")),
            "product_type": _clean(specs.get("product_type")),
            "description": _clean(p.get("description")),
            "tags": [t.strip() for t in re.split(r"[,;|]", _clean(specs.get("tags"))) if t.strip()],
            "price": _price(p.get("price")), "currency": "SGD", "inventory_qty": 1,
            "variant_grams": _clean(specs.get("weight") or specs.get("size")),
            "option": "",
            "specs_text": " | ".join(f"{k}: {_clean(v)}" for k, v in specs.items() if _clean(v)),
            "ingredients_text": _clean(specs.get("ingredients")),
            "seo_description": "",
            "source_page": p.get("source_page"),
            "extraction_mode": p.get("extraction_mode"),
            "image_description": p.get("image_description"),
        })
    return rows


# ------------------------------------------------------------------ PDF ----
TEXT_SPARSITY_THRESHOLD = 80        # below this a page is treated as image-only

PDF_SYSTEM = """You extract product data from one page of a catalogue.

For each DISTINCT product on the page return:
- "name": as written
- "price": as shown including currency symbol, or null
- "description": descriptive text for it, verbatim from the source
- "specs": attribute:value pairs actually present on the page. Use whatever keys
  fit the category. Never invent a value that is not there.
- "image_description": what a product photo shows, or null

Return strict JSON: {"products": [...]}. An empty page returns {"products": []}.
Never fabricate data that is not in the source."""


def _page_png_b64(pdf_bytes: bytes, page_number: int, zoom: float = 2.0) -> str:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[page_number].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    out = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    doc.close()
    return out


def extract_from_pdf(pdf_bytes: bytes, provider: str | None = None,
                     progress=None) -> list[dict]:
    """Text where the page has text, vision where it does not."""
    import pdfplumber
    from . import llm
    provider = provider or config.LLM_PROVIDER
    if provider == "local":
        raise RuntimeError("PDF extraction needs a model; set AGENTCARD_LLM_PROVIDER=openai")

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]

    fmt = {"type": "json_object"}
    out = []
    for i, text in enumerate(pages):
        vision = len(text.strip()) < TEXT_SPARSITY_THRESHOLD
        if progress:
            progress(i + 1, len(pages), "vision" if vision else "text")
        config.log(f"page {i+1}/{len(pages)} via {'vision' if vision else 'text'}", indent=2)
        if vision:
            res = llm.complete_vision_json(
                PDF_SYSTEM,
                f"Catalogue page {i+1}, rendered as an image because its selectable "
                f"text was too sparse to rely on. Extract the products visible on it.",
                [_page_png_b64(pdf_bytes, i)], provider=provider)
        else:
            res = llm.complete_json(PDF_SYSTEM,
                                    f"Raw text from catalogue page {i+1}:\n\n{text}",
                                    schema_format=fmt, provider=provider,
                                    model=config.CHEAP_MODEL)
        for p in (res.get("products") or []) if isinstance(res, dict) else []:
            p["source_page"] = i + 1
            p["extraction_mode"] = "vision" if vision else "text"
            out.append(p)
    return out


# ----------------------------------------------------------------- JSON ----
def _looks_shopify(item: dict) -> bool:
    return "variants" in item or "body_html" in item or ("title" in item and "id" in item)


def _shopify_row(p: dict, category: str) -> dict:
    variants = p.get("variants") or []
    prices = [v.get("price") for v in variants if v.get("price") is not None]
    price = prices[0] if prices else p.get("price")
    specs = {}
    for opt in p.get("options") or []:
        if opt.get("name") and opt.get("values"):
            specs[opt["name"]] = ", ".join(map(str, opt["values"]))
    tags = p.get("tags")
    title = str(p.get("title") or "Unnamed product")
    return {
        "id": str(p.get("handle") or re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")),
        "sku": str((variants[0] or {}).get("sku", "")) if variants else "",
        "category": category, "title": title, "brand": str(p.get("vendor") or ""),
        "product_type": str(p.get("product_type") or ""),
        "description": strip_html(p.get("body_html") or p.get("description") or ""),
        "tags": tags if isinstance(tags, list) else
                [t.strip() for t in str(tags or "").split(",") if t.strip()],
        "price": _price(price), "currency": "SGD",
        "inventory_qty": _int((variants[0] or {}).get("inventory_quantity", 1)) if variants else 1,
        "variant_grams": str((variants[0] or {}).get("grams", "")) if variants else "",
        "option": "", "specs_text": " | ".join(f"{k}: {v}" for k, v in specs.items()),
        "ingredients_text": "", "seo_description": "",
    }


def load(path: str | pathlib.Path, category: str,
         provider: str | None = None) -> tuple[list[dict], dict]:
    """Returns (rows, report). The report is what a human should review."""
    path = pathlib.Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        extracted = extract_from_pdf(path.read_bytes(), provider=provider)
        rows = rows_from_extracted(extracted, category)
        return rows, {"source": "pdf", "rows": len(rows),
                      "pages_via_vision": sum(1 for p in extracted
                                              if p.get("extraction_mode") == "vision"),
                      "mapped": {}, "unmapped": [],
                      "missing_price": sum(1 for r in rows if not r["price"]),
                      "missing_description": sum(1 for r in rows if not r["description"])}

    if suffix in (".csv", ".tsv", ".xlsx", ".xls"):
        import pandas as pd
        df = (pd.read_excel(path) if suffix in (".xlsx", ".xls")
              else pd.read_csv(path, sep="\t" if suffix == ".tsv" else ","))
        rows, report = rows_from_records(df.to_dict(orient="records"), category,
                                         [str(c) for c in df.columns])
        report["source"] = suffix.lstrip(".")
        return rows, report

    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = (raw.get("products") if isinstance(raw, dict) and isinstance(raw.get("products"), list)
                 else raw if isinstance(raw, list) else [raw])
        shopify = [i for i in items if isinstance(i, dict) and _looks_shopify(i)]
        if shopify:
            rows = [_shopify_row(p, category) for p in shopify]
            report = {"source": "shopify_json", "rows": len(rows), "mapped": {}, "unmapped": []}
        else:
            flat = [i for i in items if isinstance(i, dict)]
            headers = sorted({k for i in flat for k in i})
            rows, report = rows_from_records(flat, category, headers)
            report["source"] = "json"
        report["missing_price"] = sum(1 for r in rows if not r["price"])
        report["missing_description"] = sum(1 for r in rows if not r["description"])
        return rows, report

    raise ValueError(f"Unsupported file type: {path.suffix}")
