"""Hybrid retrieval: SQL hard filter, then embedding rank.

The order matters and is the whole argument. Semantic similarity cannot express
"under S$200" or "not for wide feet" — it can only be near them. So the typed
constraints from the Agent Card run first as SQL, eliminating products that are
actually wrong, and the embedding only ever ranks survivors.
"""
from __future__ import annotations
import json, re, sqlite3
from dataclasses import dataclass, field
import numpy as np
from . import config, embed as embedder, index as indexer

_PRICE = re.compile(r"(?:under|below|less than|max|budget(?: of)?|up to|within)\s*"
                    r"(?:s\$|sgd|\$)?\s*(\d{2,5})", re.I)
_PRICE2 = re.compile(r"(?:s\$|sgd|\$)\s*(\d{2,5})\s*(?:or less|max|budget)", re.I)

# How hard a matched situational tag pushes on the ranking.
BOOST_WEIGHT = 0.05

CALIBRATION = config.OUT / "retrieval_calibration.json"


def min_score(corpus: str = "enriched") -> float | None:
    """The score below which we decline to answer, or None if uncalibrated.

    Not a hand-picked constant: `agentcard simulate` records the similarity at
    which retrieval actually found the correct product across the whole query
    set and takes the 5th percentile. A query scoring below that is doing
    worse than the worst genuine match we have ever observed, so the honest
    answer is "nothing here fits" rather than the three least-bad shoes.

    Absolute thresholds do not transfer between embedding providers, so the
    calibration records which provider produced it and is ignored otherwise.
    """
    if not CALIBRATION.exists():
        return None
    try:
        cal = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if cal.get("embed_provider") != config.EMBED_PROVIDER:
        return None
    return cal.get(f"min_score_{corpus}")


@dataclass
class Filters:
    category: str | None = None
    price_max: float | None = None
    categorical: dict[str, str] = field(default_factory=dict)
    soft_categorical: dict[str, str] = field(default_factory=dict)
    situational: list[str] = field(default_factory=list)
    exclude_categorical: dict[str, str] = field(default_factory=dict)

    def describe(self) -> list[str]:
        out = []
        if self.category:
            out.append(f"category = {self.category}")
        if self.price_max:
            out.append(f"price <= SGD {self.price_max:.0f}")
        for k, v in self.categorical.items():
            out.append(f"{k} = {v}")
        for k, v in self.exclude_categorical.items():
            out.append(f"{k} != {v}")
        for k, v in self.soft_categorical.items():
            out.append(f"{k} ~ {v} (boost)")
        if self.situational:
            out.append("tags ∈ {" + ", ".join(self.situational) + "} (boost)")
        return out


# Only ever tested against the text BEFORE a matched phrase. Testing the phrase
# itself made "fragrance-free" read as a negation of fragrance-free, which
# returned exactly the products the shopper was trying to avoid.
# Words too generic to identify a situational tag on their own.
_TAG_STOP = {"day", "time", "under", "free", "low", "easy", "long", "use", "min",
             "week", "run", "runs", "skin", "night", "size", "office"}


def _tag_matches(tag: str, ql: str) -> bool:
    """A situational tag matches when at least half its distinctive words appear."""
    if f" {tag.replace('_', ' ')}" in ql:
        return True
    words = [w for w in tag.split("_") if len(w) >= 4 and w not in _TAG_STOP]
    if not words:
        return False
    hits = sum(1 for w in words if w in ql)
    return hits / len(words) >= 0.5


_NEGATIONS = ("not ", "no ", "without ", "avoid", "don't", "dont", "free of",
              "can't have", "cannot have", "allergic to", "reacts to")


def parse_query(q: str, configs: dict[str, dict] | None = None) -> Filters:
    """Turn shopper language into typed filters using the category configs.

    Adding a category adds its synonyms here for free — nothing in this
    function knows what a running shoe is. Longest phrase wins per key, so
    "severe overpronation" beats "pronation" rather than racing it.
    """
    configs = configs or config.all_category_configs()
    ql = f" {q.lower()} "
    f = Filters()

    m = _PRICE.search(q) or _PRICE2.search(q)
    if m:
        f.price_max = float(m.group(1))

    per_cat: dict[str, int] = {}
    best: dict[tuple[str, str], tuple[int, str, bool]] = {}   # (cat,key) -> (len,canon,neg)
    for cat, cfg in configs.items():
        hits = 0
        for key, mapping in (cfg.get("constraint_synonyms") or {}).items():
            for canon, phrases in mapping.items():
                for phrase in phrases:
                    if not phrase:
                        continue
                    pos = ql.find(f" {phrase.lower()}")
                    if pos == -1:
                        continue
                    hits += 1
                    before = ql[max(0, pos - 22):pos + 1]
                    negated = any(n in before for n in _NEGATIONS)
                    prev = best.get((cat, key))
                    if prev is None or len(phrase) > prev[0]:
                        best[(cat, key)] = (len(phrase), canon, negated)
        for tag in cfg.get("situational_vocabulary", []):
            if _tag_matches(tag, ql):
                f.situational.append(tag)
                hits += 1
        for cue in cfg.get("category_cues", []):
            if f" {cue}" in ql:
                hits += 2                     # naming the category is strong evidence
        per_cat[cat] = hits

    cat_guess = max(per_cat, key=per_cat.get) if per_cat and max(per_cat.values()) else None
    f.category = cat_guess

    for (cat, key), (_, canon, negated) in best.items():
        if cat_guess and cat != cat_guess:
            continue
        hard = key in set(configs[cat].get("hard_filter_keys", []))
        if negated:
            f.exclude_categorical[key] = canon
        elif hard:
            f.categorical[key] = canon
        else:
            f.soft_categorical[key] = canon
    f.situational = sorted(set(f.situational))
    return f


def sql_filter(f: Filters, db=None, in_stock_only: bool = True) -> list[str]:
    con = sqlite3.connect(db or indexer.DB)
    sql = "SELECT id FROM products WHERE 1=1"
    args: list = []
    if f.category:
        sql += " AND category = ?"
        args.append(f.category)
    if f.price_max:
        sql += " AND price <= ?"
        args.append(f.price_max)
    if in_stock_only:
        sql += " AND availability != 'out_of_stock'"
    for k, v in f.categorical.items():
        sql += (" AND id IN (SELECT product_id FROM constraints "
                "WHERE key=? AND value=?)")
        args += [k, str(v).lower()]
    for k, v in f.exclude_categorical.items():
        sql += (" AND id NOT IN (SELECT product_id FROM constraints "
                "WHERE key=? AND value=?)")
        args += [k, str(v).lower()]
    ids = [r[0] for r in con.execute(sql, args)]
    con.close()
    return ids


def categorical_boost(soft: dict[str, str], db=None) -> dict[str, float]:
    """Preference constraints rank rather than eliminate."""
    if not soft:
        return {}
    con = sqlite3.connect(db or indexer.DB)
    out: dict[str, float] = {}
    for key, val in soft.items():
        for (pid,) in con.execute(
                "SELECT product_id FROM constraints WHERE key=? AND value=?",
                (key, str(val).lower())):
            out[pid] = out.get(pid, 0.0) + 1.0
    con.close()
    return out


def combined_boost(f: "Filters", db=None) -> dict[str, float]:
    a = situational_boost(f.situational, db)
    b = categorical_boost(f.soft_categorical, db)
    for pid, v in b.items():
        a[pid] = a.get(pid, 0.0) + v
    return a


def situational_boost(tags: list[str], db=None) -> dict[str, float]:
    """Soft signal, not a filter.

    A shopper saying "humid Singapore weather" should not have products
    eliminated for lacking a humid_climate tag — enrichment coverage varies. So
    matched situational tags nudge ranking instead of gating it.
    """
    if not tags:
        return {}
    con = sqlite3.connect(db or indexer.DB)
    q = ("SELECT product_id, COUNT(*) FROM constraints WHERE key='situational_tag' "
         "AND value IN (%s) GROUP BY product_id" % ",".join("?" * len(tags)))
    out = {pid: float(n) for pid, n in con.execute(q, [t.lower() for t in tags])}
    con.close()
    return out


def _relax(f: Filters) -> list[Filters]:
    """Ordered fallbacks, widest constraint dropped last."""
    out = []
    if f.categorical:
        out.append(Filters(f.category, f.price_max, {}, f.soft_categorical,
                           f.situational, f.exclude_categorical))
    if f.price_max:
        out.append(Filters(f.category, None, {}, f.soft_categorical, f.situational, {}))
    out.append(Filters(f.category))
    out.append(Filters())
    return out


def search(query: str, k: int = 3, corpus: str = "enriched",
           use_filter: bool = True, embed_provider: str | None = None,
           configs: dict | None = None) -> dict:
    ids, vpage, vcard = indexer.load()
    pos = {pid: i for i, pid in enumerate(ids)}

    f = parse_query(query, configs)
    relaxed_by = None
    if use_filter:
        candidates = sql_filter(f)
        if len(candidates) < k:
            for g in _relax(f):
                candidates = sql_filter(g)
                relaxed_by = "constraints relaxed to widen the pool"
                if len(candidates) >= k:
                    f = g
                    break
    else:
        candidates = list(ids)
    if not candidates:
        candidates = list(ids)

    qv = embedder.embed([query], provider=embed_provider)[0]
    pool = [c for c in candidates if c in pos]
    scores = indexer.scores(qv, vpage, vcard, [pos[c] for c in pool], corpus)
    boost = combined_boost(f)
    scores = scores + np.array([BOOST_WEIGHT * boost.get(c, 0.0) for c in pool],
                               dtype=np.float32)
    order = np.argsort(-scores)[:k]
    hits = [{"id": pool[int(o)], "score": float(scores[int(o)])} for o in order]

    floor = min_score(corpus)
    best = float(scores.max()) if len(scores) else 0.0
    return {"query": query, "corpus": corpus, "filters": f, "filtered_to": len(candidates),
            "relaxed": relaxed_by, "hits": hits, "best_score": round(best, 4),
            "floor": floor, "no_good_match": bool(floor is not None and best < floor)}


def hydrate(hits: list[dict]) -> list[dict]:
    cards = json.loads((config.OUT / "agent_cards.json").read_text(encoding="utf-8"))
    out = []
    for h in hits:
        c = cards.get(h["id"], {})
        out.append({**h, "card": c,
                    "title": c.get("identity", {}).get("title", h["id"]),
                    "price": (c.get("identity", {}).get("price") or {}).get("amount"),
                    "readiness": c.get("readiness", {}).get("score")})
    return out
