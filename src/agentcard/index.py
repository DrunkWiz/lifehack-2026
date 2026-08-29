"""Build the two corpora and the constraint store.

Each product gets TWO vectors: its page text, and its Agent Card. A product's
score for a query is the better of the two.

This is the second attempt at the join and the reason for it matters. Indexing
the card *instead of* the page threw away the brand's own copy, so enrichment
could only lose information. Concatenating them was worse in a subtler way:
embeddings are normalised over the whole document, so bolting 2,600 characters
of card onto 500 characters of page dilutes a page that already matched — which
is exactly why skincare scored lower enriched than raw.

Two vectors, take the max. Standard multi-field retrieval, and it says the true
thing: the card is an additional document about the product, not a replacement
for the page and not an extension of it.

Note this does NOT guarantee better recall. Every product gains a card, so a
rival's card can outrank yours; only the per-product score is monotone, not the
ranking. The measurement stays honest.

The constraint store is a plain SQLite table. Not because SQLite is exciting,
but because 'the hard filter is SQL a brand's existing stack already speaks' is
the integration story, and a vector database would obscure it.
"""
from __future__ import annotations
import json, sqlite3
import numpy as np
from . import config, embed as embedder, ingest, schema

DB = config.OUT / "catalogue.db"
VEC_PAGE = config.OUT / "vectors_page.npy"
VEC_CARD = config.OUT / "vectors_card.npy"
IDS = config.OUT / "corpus_ids.json"


def card_text(card: dict) -> str:
    """Flatten an Agent Card into the text an embedding should see."""
    idy = card.get("identity", {})
    parts = [idy.get("title", ""), idy.get("brand", ""),
             card.get("narrative", {}).get("one_line_pitch", "")]
    for u in card.get("use_cases", []):
        parts.append(f'{u.get("scenario","")}. {u.get("why_it_fits","")}')
    for p in card.get("personas", []):
        parts.append(f'{p.get("fit","")} fit for {p.get("label","")}: {p.get("reasoning","")}')
    for n in card.get("not_for", []):
        parts.append(f'Not for {n.get("exclusion","")} — {n.get("reason","")}')
    for c in card.get("comparisons", []):
        parts.append(f'{c.get("direction","")} {c.get("axis","")} than {c.get("against","")} '
                     f'({c.get("magnitude","")}); tradeoff: {c.get("tradeoff","")}')
    for v in card.get("narrative", {}).get("intent_variants", []):
        parts.append(v.get("copy", ""))
    parts += [t.replace("_", " ") for t in schema.situational_tags(card)]
    for n in card.get("hard_constraints", {}).get("numeric", []):
        parts.append(f'{n.get("key","").replace("_"," ")} {n.get("value")}{n.get("unit","")}')
    for c in card.get("hard_constraints", {}).get("categorical", []):
        parts.append(f'{c.get("key","").replace("_"," ")}: {", ".join(map(str, c.get("values", [])))}')
    return " \n".join(p for p in parts if p)


def build(embed_provider: str | None = None) -> dict:
    rows = {r["id"]: r for r in ingest.load_all()}
    cards = json.loads((config.OUT / "agent_cards.json").read_text(encoding="utf-8"))
    ids = [i for i in rows if i in cards]
    missing = [i for i in rows if i not in cards]
    config.log(f"{len(ids)} products with cards"
               + (f" ({len(missing)} rows have no card yet — run enrich)" if missing else ""))

    page_docs = [ingest.raw_text(rows[i]) for i in ids]
    card_docs = [card_text(cards[i]) for i in ids]

    config.log("page vectors (what a crawler reads today)")
    np.save(VEC_PAGE, embedder.embed(page_docs, provider=embed_provider))
    config.log("card vectors (the Agent Card, indexed separately)")
    np.save(VEC_CARD, embedder.embed(card_docs, provider=embed_provider))
    IDS.write_text(json.dumps(ids), encoding="utf-8")
    config.log(f"mean document length: page {sum(map(len, page_docs)) // max(len(ids), 1)} "
               f"chars, card {sum(map(len, card_docs)) // max(len(ids), 1)} chars")

    config.log("building the constraint store")

    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript("""
      CREATE TABLE products(
        id TEXT PRIMARY KEY, title TEXT, brand TEXT, category TEXT,
        price REAL, currency TEXT, availability TEXT, readiness REAL,
        readiness_raw REAL, pitch TEXT);
      CREATE TABLE constraints(
        product_id TEXT, kind TEXT, key TEXT, value TEXT, num REAL);
      CREATE INDEX ix_c ON constraints(key, value);
      CREATE INDEX ix_cp ON constraints(product_id);
    """)
    baselines = json.loads((config.OUT / "raw_baseline_cards.json").read_text(encoding="utf-8"))
    for i in ids:
        c, r = cards[i], rows[i]
        idy = c.get("identity", {})
        con.execute("INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?,?)", (
            i, idy.get("title", r["title"]), idy.get("brand", r["brand"]),
            idy.get("category", r["category"]),
            (idy.get("price") or {}).get("amount", r["price"]), "SGD",
            idy.get("availability", "in_stock"),
            c.get("readiness", {}).get("score", 0.0),
            baselines.get(i, {}).get("readiness", {}).get("score", 0.0),
            c.get("narrative", {}).get("one_line_pitch", "")))
        for n in c.get("hard_constraints", {}).get("numeric", []):
            con.execute("INSERT INTO constraints VALUES (?,?,?,?,?)",
                        (i, "numeric", n.get("key"), str(n.get("value")), float(n.get("value", 0))))
        for cc in c.get("hard_constraints", {}).get("categorical", []):
            for v in cc.get("values", []):
                con.execute("INSERT INTO constraints VALUES (?,?,?,?,?)",
                            (i, "categorical", cc.get("key"), str(v).lower(), None))
        for t in schema.situational_tags(c):
            con.execute("INSERT INTO constraints VALUES (?,?,?,?,?)",
                        (i, "situational", "situational_tag", t.lower(), None))
    n_con, n_tag = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT CASE WHEN kind='situational' THEN value END) "
        "FROM constraints").fetchone()
    con.commit()
    con.close()
    config.log(f"{n_con} filterable constraints, {n_tag} distinct situational tags")
    return {"indexed": len(ids), "constraints": n_con, "situational_tags": n_tag,
            "db": str(DB)}


def load():
    """(ids, page_vectors, card_vectors)."""
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    return ids, np.load(VEC_PAGE), np.load(VEC_CARD)


def scores(qv, page, card, idxs, corpus: str = "enriched"):
    """Similarity for a set of rows.

    corpus="raw"      the page alone — what an agent can read today
    corpus="enriched" the better of page and card, because a product is
                      findable if EITHER document answers the query
    """
    import numpy as _np
    p = page[idxs] @ qv
    if corpus == "raw":
        return p
    return _np.maximum(p, card[idxs] @ qv)
