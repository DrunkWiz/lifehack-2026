"""
Extracts a normalized list of product dicts from an uploaded catalog file.

Supported inputs:
- CSV / XLSX: parsed directly with pandas (structured already).
- JSON: either a flat list/array of product objects, or Shopify's nested
  `products.json` shape (title, body_html, variants, options, tags, images).
- PDF: per page, try text extraction first (pdfplumber). If a page's extracted
  text is too sparse (scanned page, brochure/lookbook with little selectable
  text), fall back to rendering that page as an image and using a vision-
  capable model to read it directly.

Every product dict has the shape:
{
    "name": str,
    "price": str | None,
    "description": str,
    "specs": dict,              # whatever attribute:value pairs were found
    "image_description": str | None,
    "source_page": int | None,
}
"""

import io
import re
import json
import base64
import pandas as pd
import pdfplumber
import fitz  # PyMuPDF

from llm_utils import call_llm_json, call_llm_vision_json

TEXT_SPARSITY_THRESHOLD = 80  # chars; below this we treat a PDF page as image-only

EXTRACTION_SYSTEM_PROMPT = """You are an expert catalog data extraction assistant.
Given raw content from ONE page of a product catalog (as text, or as an image),
identify each DISTINCT product mentioned on it.

For each product return:
- "name": product name as written
- "price": price if found (string, keep currency symbol as shown), else null
- "description": any descriptive/marketing text found for it (verbatim from source)
- "specs": a JSON object of attribute:value pairs you can actually find on the page
  (e.g. weight, material, color, size, skin_type, ingredients — use whatever keys fit
  the product category). Do NOT invent values that are not present.
- "image_description": if a product image/photo is visible, briefly describe what it shows
  (colorway, setting, styling). If no image, use null.

Return strict JSON: {"products": [ {...}, ... ]}
If no products are found on this page, return {"products": []}.
Never fabricate data that isn't present in the source."""


def render_page_as_base64(pdf_bytes: bytes, page_number: int, zoom: float = 2.0) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode("utf-8")


def extract_text_pages(pdf_bytes: bytes) -> list[str]:
    pages_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")
    return pages_text


def extract_products_from_pdf(pdf_bytes: bytes, progress_callback=None) -> list[dict]:
    pages_text = extract_text_pages(pdf_bytes)
    all_products = []
    total = len(pages_text)

    for i, text in enumerate(pages_text):
        used_vision = len(text.strip()) < TEXT_SPARSITY_THRESHOLD
        if progress_callback:
            progress_callback(i + 1, total, "vision" if used_vision else "text")

        if not used_vision:
            result = call_llm_json(
                EXTRACTION_SYSTEM_PROMPT,
                f"Raw text extracted from catalog page {i + 1}:\n\n{text}",
            )
        else:
            b64 = render_page_as_base64(pdf_bytes, i)
            result = call_llm_vision_json(
                EXTRACTION_SYSTEM_PROMPT,
                f"This is catalog page {i + 1}, rendered as an image "
                f"(selectable text on this page was too sparse to rely on). "
                f"Extract the products visible on it.",
                [b64],
            )

        products = result.get("products", []) if isinstance(result, dict) else []
        for p in products:
            p.setdefault("price", None)
            p.setdefault("description", "")
            p.setdefault("specs", {})
            p.setdefault("image_description", None)
            p["source_page"] = i + 1
            p["extraction_mode"] = "vision" if used_vision else "text"
        all_products.extend(products)

    return all_products


# Real exports rarely name the column "price" - Shopify uses "Variant Price", others use
# "Unit Price" or "MSRP". Matching only the exact word leaves price null on most real files,
# which silently breaks every budget-constrained query ("under S$200").
PRICE_KEYS = ("price", "variant price", "unit price", "sale price", "retail price", "cost", "msrp")
NAME_KEYS = ("name", "product", "product name", "title")
DESC_KEYS = ("description", "desc", "body (html)", "body html", "body")


def _clean_price(value) -> str | None:
    """Keep the currency symbol if one is present, drop stray whitespace, reject junk."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return None
    return text if re.search(r"\d", text) else None


def extract_products_from_table(df: pd.DataFrame) -> list[dict]:
    records = df.to_dict(orient="records")
    reserved_name_keys = set(NAME_KEYS)
    reserved_price_keys = set(PRICE_KEYS)
    reserved_desc_keys = set(DESC_KEYS)

    products = []
    for r in records:
        lower_map = {str(k).strip().lower(): k for k in r.keys()}

        name_key = next((lower_map[k] for k in NAME_KEYS if k in lower_map), None)
        price_key = next((lower_map[k] for k in PRICE_KEYS if k in lower_map), None)
        desc_key = next((lower_map[k] for k in DESC_KEYS if k in lower_map), None)

        name = r.get(name_key) if name_key else str(list(r.values())[0])
        price = _clean_price(r.get(price_key)) if price_key else None
        desc = r.get(desc_key) if desc_key else ""
        if desc and "<" in str(desc):
            desc = _strip_html(str(desc))

        used_keys = {k for k in (name_key, price_key, desc_key) if k}
        specs = {
            k: v for k, v in r.items()
            if k not in used_keys and v is not None and str(v).strip() not in ("", "nan")
        }

        products.append({
            "name": str(name) if name is not None else "Unnamed product",
            "price": str(price) if price is not None else None,
            "description": str(desc) if desc else "",
            "specs": {str(k): str(v) for k, v in specs.items()},
            "image_description": None,
            "source_page": None,
            "extraction_mode": "table",
        })
    return products


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&amp;|&quot;|&#39;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_shopify_product(item: dict) -> bool:
    return "variants" in item or "body_html" in item or ("title" in item and "id" in item)


def _shopify_product_to_dict(p: dict) -> dict:
    name = p.get("title") or "Unnamed product"
    variants = p.get("variants") or []

    price = None
    if variants:
        prices = [v.get("price") for v in variants if v.get("price") is not None]
        if len(set(prices)) > 1:
            try:
                nums = sorted(float(x) for x in prices)
                price = f"{nums[0]} - {nums[-1]}"
            except (TypeError, ValueError):
                price = prices[0]
        elif prices:
            price = prices[0]
    elif p.get("price") is not None:
        price = p.get("price")

    description = _strip_html(p.get("body_html") or p.get("description") or "")

    specs = {}
    if p.get("vendor"):
        specs["vendor"] = str(p["vendor"])
    if p.get("product_type"):
        specs["product_type"] = str(p["product_type"])

    tags = p.get("tags")
    if tags:
        specs["tags"] = ", ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags)

    for opt in p.get("options") or []:
        oname, ovalues = opt.get("name"), opt.get("values")
        if oname and ovalues:
            specs[oname] = ", ".join(str(v) for v in ovalues)

    image_description = None
    images = p.get("images") or []
    if images:
        alt_texts = [img.get("alt") for img in images if img.get("alt")]
        image_description = "; ".join(alt_texts[:3]) if alt_texts else f"{len(images)} product image(s) present"

    return {
        "name": str(name),
        "price": str(price) if price is not None else None,
        "description": description,
        "specs": specs,
        "image_description": image_description,
        "source_page": None,
        "extraction_mode": "json_shopify",
    }


def _generic_json_product_to_dict(item: dict) -> dict:
    lower_map = {str(k).strip().lower(): k for k in item.keys()}
    name_key = next((lower_map[k] for k in ("name", "product", "product name", "title") if k in lower_map), None)
    price_key = next((lower_map[k] for k in ("price",) if k in lower_map), None)
    desc_key = next((lower_map[k] for k in ("description", "desc") if k in lower_map), None)

    name = item.get(name_key) if name_key else str(next(iter(item.values()), "Unnamed product"))
    price = item.get(price_key) if price_key else None
    desc = item.get(desc_key) if desc_key else ""

    used_keys = {k for k in (name_key, price_key, desc_key) if k}
    specs = {}
    for k, v in item.items():
        if k in used_keys or v is None:
            continue
        if isinstance(v, (dict, list)):
            specs[str(k)] = json.dumps(v) if v else ""
        else:
            specs[str(k)] = str(v)

    return {
        "name": str(name) if name is not None else "Unnamed product",
        "price": str(price) if price is not None else None,
        "description": str(desc) if desc else "",
        "specs": specs,
        "image_description": None,
        "source_page": None,
        "extraction_mode": "json_generic",
    }


def extract_products_from_json(uploaded_file) -> list[dict]:
    raw = json.load(uploaded_file)

    if isinstance(raw, dict) and isinstance(raw.get("products"), list):
        items = raw["products"]
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    else:
        items = []

    products = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if _looks_like_shopify_product(item):
            products.append(_shopify_product_to_dict(item))
        else:
            products.append(_generic_json_product_to_dict(item))
    return products


def load_catalog_file(uploaded_file, progress_callback=None) -> list[dict]:
    """Dispatch based on file extension. Returns a flat list of raw product dicts."""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        pdf_bytes = uploaded_file.read()
        return extract_products_from_pdf(pdf_bytes, progress_callback=progress_callback)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return extract_products_from_table(df)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        return extract_products_from_table(df)
    elif name.endswith(".json"):
        return extract_products_from_json(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")
