"""The loader is the adoptability story, so it gets tested like one.

Every failure here is a brand uploading their real export and seeing an empty
catalogue — which is exactly what the exact-match column mapping in the earlier
prototype did to a Shopify CSV.
"""
import json, pathlib, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentcard import catalog_loader as cl  # noqa: E402


def test_shopify_export_columns_are_mapped():
    """Variant Price and Body (HTML), not price and description."""
    headers = ["Handle", "Title", "Body (HTML)", "Vendor", "Type", "Tags", "Published",
               "Variant SKU", "Variant Grams", "Variant Inventory Qty", "Variant Price",
               "Image Src", "SEO Description"]
    cmap, unmapped = cl.map_columns(headers)
    assert cmap["price"] == "Variant Price"
    assert cmap["description"] == "Body (HTML)"
    assert cmap["title"] == "Title"
    assert cmap["brand"] == "Vendor"
    assert cmap["sku"] == "Variant SKU"


@pytest.mark.parametrize("headers,field,expected", [
    (["Product Name", "Unit Price"], "price", "Unit Price"),
    (["name", "RRP"], "price", "RRP"),
    (["Item Name", "Long Description"], "description", "Long Description"),
    (["title", "Manufacturer"], "brand", "Manufacturer"),
    (["title", "Stock"], "inventory", "Stock"),
])
def test_common_export_dialects(headers, field, expected):
    cmap, _ = cl.map_columns(headers)
    assert cmap.get(field) == expected


def test_a_header_is_never_claimed_twice():
    headers = ["Title", "SEO Title", "Description", "SEO Description"]
    cmap, _ = cl.map_columns(headers)
    assert len(set(cmap.values())) == len(cmap)


def test_exact_match_beats_substring():
    cmap, _ = cl.map_columns(["Compare At Price", "Price"])
    assert cmap["price"] == "Price"


def test_html_is_stripped_not_dumped_into_specs(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("Title,Body (HTML),Variant Price\n"
                   'Widget,"<p>Soft &amp; light</p>",19.00\n', encoding="utf-8")
    rows, rep = cl.load(csv, "test.cat")
    assert rows[0]["description"] == "Soft & light"
    assert "<p>" not in rows[0]["specs_text"]
    assert rep["missing_price"] == 0 and rep["missing_description"] == 0


def test_unmapped_columns_are_kept_as_readable_specs(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("Title,Variant Price,Thread Count,Certification\n"
                   "Sheet,49.00,400,OEKO-TEX\n", encoding="utf-8")
    rows, rep = cl.load(csv, "test.cat")
    assert "Thread Count: 400" in rows[0]["specs_text"]
    assert "Certification: OEKO-TEX" in rows[0]["specs_text"]
    assert set(rep["unmapped"]) == {"Thread Count", "Certification"}


def test_prices_with_currency_symbols_and_commas(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("Title,Price\nA,\"S$1,299.00\"\nB,€89\nC,\nD,POA\n", encoding="utf-8")
    rows, rep = cl.load(csv, "test.cat")
    assert [r["price"] for r in rows] == [1299.0, 89.0, 0.0, 0.0]
    assert rep["missing_price"] == 2      # reported, not silently zeroed


def test_shopify_products_json(tmp_path):
    p = tmp_path / "products.json"
    p.write_text(json.dumps({"products": [{
        "id": 1, "title": "Aero 9", "handle": "aero-9", "vendor": "Vela",
        "body_html": "<p>Fast.</p>", "tags": ["road"],
        "variants": [{"price": "199.00", "sku": "A9", "grams": 236,
                      "inventory_quantity": 12}],
        "options": [{"name": "Size", "values": ["US M9"]}]}]}), encoding="utf-8")
    rows, rep = cl.load(p, "test.shoes")
    assert rep["source"] == "shopify_json"
    assert rows[0]["price"] == 199.0 and rows[0]["brand"] == "Vela"
    assert rows[0]["description"] == "Fast."


def test_rows_are_shaped_like_the_pipeline_expects():
    from agentcard import ingest
    native = ingest.load_all()[0]
    loaded, _ = cl.load(ROOT / "data" / "raw" / "spec_rich" / "shopify_running_shoes.csv",
                        "footwear.running")
    assert set(native) <= set(loaded[0]), "loader output is missing pipeline fields"


def test_pdf_path_refuses_the_offline_provider():
    with pytest.raises(RuntimeError, match="needs a model"):
        cl.extract_from_pdf(b"%PDF-1.4", provider="local")
