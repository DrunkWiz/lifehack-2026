"""
Retrieval layer — the part that answers "would an AI assistant actually pick this product?"

The rest of the app PRODUCES agent-optimized content. This module TESTS it: a shopper's
natural-language query is run against the catalog twice — once over the raw catalog text
as extracted, once over the generated content — so the two ranked result sets can be shown
side by side.

Deliberately no vector database. At catalog scale (hundreds to low thousands of products)
a dot product over a row-normalized embedding matrix is sub-millisecond, needs no extra
dependency beyond numpy, and keeps the Streamlit Cloud image small and the cold start fast.
faiss/chroma would add install weight and boot latency for no measurable gain here.
"""

import math
import numpy as np

from llm_utils import embed_texts, call_llm_json

TOP_K = 5


def _s(val, default: str = "") -> str:
    """Safely coerce any field (incl. NaN floats from pandas, None) to a string."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return str(val)


# ---------------------------------------------------------------------------
# Text views of a product — the two things we compare
# ---------------------------------------------------------------------------

def product_to_raw_text(product: dict) -> str:
    """The catalog as the brand actually supplied it. This is the 'before'."""
    specs = product.get("specs") or {}
    spec_str = "; ".join(f"{k}: {v}" for k, v in specs.items() if _s(v).strip())
    parts = [
        _s(product.get("name"), "Unnamed product"),
        f"Price: {_s(product.get('price'), 'not stated')}",
        _s(product.get("description")),
        spec_str,
    ]
    return "\n".join(p for p in parts if p.strip())


def product_to_optimized_text(product: dict, generated: str) -> str:
    """Generated content, kept with name and price so both sides can match on identity
    and budget. Without these the comparison would be unfair in the other direction.

    Normalized attributes are appended too: the prose is written for one persona, so a query
    from a different angle ("wide fit", "stability") would otherwise have nothing to match on
    even though the data is right there. This is what stops persona-targeted copy from
    degrading retrieval for every other intent."""
    normalized = product.get("specs_normalized") or {}
    parts = [
        _s(product.get("name"), "Unnamed product"),
        f"Price: {_s(product.get('price'), 'not stated')}",
        _s(generated),
    ]
    if normalized:
        parts.append("Attributes: " + "; ".join(
            f"{k}: {v}" for k, v in normalized.items() if v is not None and _s(v).strip()))
    return "\n".join(p for p in parts if p.strip())


# ---------------------------------------------------------------------------
# Index build + search
# ---------------------------------------------------------------------------

def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def build_index(entries: list[dict]) -> dict:
    """entries: [{"key": str, "name": str, "cluster": str, "text": str}, ...]

    Returns {"matrix": np.ndarray | None, "entries": [...]}. Rows are L2-normalized,
    so cosine similarity is a plain dot product at query time."""
    entries = [e for e in entries if _s(e.get("text")).strip()]
    if not entries:
        return {"matrix": None, "entries": []}
    vectors = embed_texts([e["text"] for e in entries])
    matrix = _normalize(np.array(vectors, dtype=np.float32))
    return {"matrix": matrix, "entries": entries}


def search(index: dict, query_vec: np.ndarray, k: int = TOP_K) -> list[dict]:
    """Cosine similarity against a normalized matrix. Returns ranked hits."""
    if index.get("matrix") is None or not index.get("entries"):
        return []
    sims = index["matrix"] @ query_vec
    k = min(k, len(sims))
    top = np.argsort(-sims)[:k]
    return [
        {
            "rank": position + 1,
            "score": round(float(sims[row]), 4),
            "key": index["entries"][row]["key"],
            "name": index["entries"][row]["name"],
            "cluster": index["entries"][row].get("cluster", ""),
            "text": index["entries"][row]["text"],
        }
        for position, row in enumerate(top)
    ]


def embed_query(query: str) -> np.ndarray:
    vec = np.array(embed_texts([query])[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


# ---------------------------------------------------------------------------
# Rerank — the step that makes the result explainable rather than a cosine score
# ---------------------------------------------------------------------------

RERANK_SYSTEM_PROMPT = """You are the recommendation step of an AI shopping assistant.
A shopper has stated what they need. You are given candidate products and, for each, the
ONLY content available about it.

Decide which you would actually recommend, in order.

Hard rules:
- Justify a pick ONLY with information actually present in that candidate's content.
  Never infer, estimate, or invent a spec, number, material or claim.
- Judge purely on the content given. Ignore anything you may know about these brands
  or products from elsewhere — if the content does not say it, it is not available.
- If a candidate cannot be justified against what the shopper asked for, still rank it,
  set "recommend" to false, and say plainly which of the shopper's stated needs the
  content fails to answer.

Return strict JSON:
{"ranked": [
  {"key": "<exactly the candidate key given to you>",
   "recommend": true or false,
   "reason": "one sentence citing the specific attributes and values that justified this",
   "cited_attributes": ["attribute name or value quoted from the content", ...],
   "unanswered": ["what the shopper asked about that this content does not address", ...]}
]}
Rank every candidate you were given, best first."""


def rerank(query: str, hits: list[dict]) -> list[dict]:
    """Ask the model to order the retrieved candidates and cite what justified each pick.

    Retrieval alone yields cosine scores, which say nothing about WHY a product matched.
    This step is what turns the demo from 'here are some vectors' into 'here is the
    attribute that won it, and here is what the content still cannot answer'."""
    if not hits:
        return []

    candidates = "\n\n".join(
        f"[key: {h['key']}]\nName: {h['name']}\nContent on file:\n{h['text'][:1200]}"
        for h in hits
    )
    result = call_llm_json(
        RERANK_SYSTEM_PROMPT,
        f"Shopper's request:\n{query}\n\nCandidate products:\n\n{candidates}",
        temperature=0.1,
    )
    ranked = result.get("ranked", []) if isinstance(result, dict) else []

    by_key = {h["key"]: h for h in hits}
    ordered = []
    seen = set()
    for position, item in enumerate(ranked):
        key = item.get("key")
        if key not in by_key or key in seen:
            continue  # guard against hallucinated or duplicated keys
        seen.add(key)
        hit = dict(by_key[key])
        hit.update({
            "final_rank": position + 1,
            "recommend": bool(item.get("recommend", False)),
            "reason": _s(item.get("reason")),
            "cited_attributes": [_s(a) for a in (item.get("cited_attributes") or [])],
            "unanswered": [_s(a) for a in (item.get("unanswered") or [])],
        })
        ordered.append(hit)

    # anything the model dropped stays visible, in retrieval order, rather than vanishing
    for hit in hits:
        if hit["key"] not in seen:
            extra = dict(hit)
            extra.update({
                "final_rank": len(ordered) + 1,
                "recommend": False,
                "reason": "Not ranked by the assistant.",
                "cited_attributes": [],
                "unanswered": [],
            })
            ordered.append(extra)
    return ordered


def run_query(raw_index: dict, optimized_index: dict, query: str, k: int = TOP_K) -> dict:
    """Embed the query once, search both indexes, rerank both. Returns the before/after pair
    plus the rank movement per product that appears in both."""
    query_vec = embed_query(query)

    raw_hits = rerank(query, search(raw_index, query_vec, k))
    optimized_hits = rerank(query, search(optimized_index, query_vec, k))

    raw_rank = {h["key"]: h["final_rank"] for h in raw_hits}
    movement = []
    for hit in optimized_hits:
        before = raw_rank.get(hit["key"])
        movement.append({
            "name": hit["name"],
            "before": before,          # None = did not surface at all on raw content
            "after": hit["final_rank"],
        })

    return {"query": query, "raw": raw_hits, "optimized": optimized_hits, "movement": movement}
