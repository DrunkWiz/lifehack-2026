"""
Data request: what the brand has to go and measure.

The rest of the app can only rewrite what the catalog already contains - generation is
grounded in verified attributes precisely so it cannot invent a breathability rating that
nobody supplied. Which means for genuinely absent data, better copy is impossible, and the
only useful output is a list handed to whoever owns the product data.

Two views:
  attribute_requests() - by attribute, ranked by how many products are missing it, annotated
                         with the persona criterion that needed it. This is the work order.
  product_requests()   - by product, for anyone fixing one SKU at a time.
"""

from collections import defaultdict


def _s(val, default: str = "") -> str:
    return default if val is None else str(val)


def _attribute_applies(product: dict, attribute: str) -> bool:
    if "applicable_attributes" not in product:
        return True
    return attribute in (product.get("applicable_attributes") or [])


def attribute_requests(cluster_data: dict) -> list[dict]:
    """One row per (cluster, attribute) that is missing somewhere, most-missing first."""
    rows = []
    for cluster_name, data in cluster_data.items():
        members = data.get("members", [])
        total = len(members)
        if not total:
            continue
        schema = data.get("expected_attrs", []) or []

        # why each attribute matters: the persona criteria that depend on it
        reasons = defaultdict(list)
        for persona in data.get("personas", []) or []:
            pf = persona.get("fit") or {}
            for criterion in pf.get("criteria", []) or []:
                rationale = criterion.get("rationale") or ""
                if rationale:
                    reasons[criterion["attribute"]].append(f"{persona.get('title','')}: {rationale}")

        for attr in schema:
            applicable_members = [p for p in members if _attribute_applies(p, attr)]
            if not applicable_members:
                continue
            missing = [
                _s(p.get("name"), "Unnamed product") for p in applicable_members
                if not _s((p.get("specs_normalized") or {}).get(attr)).strip()
            ]
            if not missing:
                continue
            rows.append({
                "cluster": cluster_name,
                "attribute": attr,
                "missing_count": len(missing),
                "total_products": len(applicable_members),
                "catalog_products": total,
                "applicable_products": len(applicable_members),
                "missing_pct": round(100 * len(missing) / len(applicable_members), 1),
                "needed_because": " | ".join(dict.fromkeys(reasons.get(attr, []))) or
                                  "Expected for this category by buyers and shopping agents",
                "missing_products": ", ".join(missing),
            })
    rows.sort(key=lambda r: (-r["missing_count"], r["cluster"], r["attribute"]))
    return rows


def product_requests(cluster_data: dict) -> list[dict]:
    """One row per product that is missing anything, most-incomplete first."""
    rows = []
    for cluster_name, data in cluster_data.items():
        schema = data.get("expected_attrs", []) or []
        if not schema:
            continue
        for product in data.get("members", []):
            normalized = product.get("specs_normalized") or {}
            applicable = [a for a in schema if _attribute_applies(product, a)]
            missing = [a for a in applicable if not _s(normalized.get(a)).strip()]
            if not missing:
                continue
            rows.append({
                "cluster": cluster_name,
                "product": _s(product.get("name"), "Unnamed product"),
                "missing_count": len(missing),
                "schema_size": len(applicable),
                "completeness_pct": round(100 * (len(applicable) - len(missing)) /
                                          len(applicable), 1) if applicable else 100.0,
                "missing_attributes": ", ".join(missing),
                "not_applicable_attributes": ", ".join(a for a in schema if a not in applicable),
            })
    rows.sort(key=lambda r: (-r["missing_count"], r["cluster"], r["product"]))
    return rows


def headline(cluster_data: dict) -> dict:
    """Numbers for the summary line above the table."""
    attr_rows = attribute_requests(cluster_data)
    prod_rows = product_requests(cluster_data)
    total_products = sum(len(d.get("members", [])) for d in cluster_data.values())
    return {
        "attributes_needed": len(attr_rows),
        "products_affected": len(prod_rows),
        "total_products": total_products,
        "worst": attr_rows[0] if attr_rows else None,
    }
