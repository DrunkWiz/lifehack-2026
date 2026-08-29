"""Embeddings with a local fallback.

provider=openai -> text-embedding-3-small, disk-cached by sha256(text)
provider=local  -> deterministic hashed TF-IDF over word unigrams and bigrams,
                   with synonym expansion driven by the category config.

The local embedder is lexical, not semantic. It is honest about that: it exists
so the pipeline, the simulator and the UI all run with no network. Numbers from
a local run should be read as a lower bound on the architecture, not as the
result you quote to a judge.
"""
from __future__ import annotations
import hashlib, json, re, math
import numpy as np
from . import config, llm

DIM = 2048
_CACHE = config.CACHE / "embeddings"
_CACHE.mkdir(parents=True, exist_ok=True)
_WORD = re.compile(r"[a-z0-9]+")

_SYN: dict[str, list[str]] = {}


def load_synonyms() -> dict[str, list[str]]:
    """phrase -> canonical tokens, built from every category config."""
    global _SYN
    if _SYN:
        return _SYN
    syn: dict[str, list[str]] = {}
    for cfg in config.all_category_configs().values():
        for key, mapping in (cfg.get("constraint_synonyms") or {}).items():
            for canon, phrases in mapping.items():
                for p in phrases:
                    syn.setdefault(p.lower(), []).extend([f"{key}_{canon}", canon.lower()])
        for tag in cfg.get("situational_vocabulary", []):
            syn.setdefault(tag.replace("_", " "), []).append(tag)
    _SYN = syn
    return syn


def _tokens(text: str) -> list[str]:
    t = text.lower()
    extra = [tok for phrase, toks in load_synonyms().items() if phrase in t for tok in toks]
    words = _WORD.findall(t) + extra
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return grams


def _local_vector(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    toks = _tokens(text)
    if not toks:
        return v
    counts: dict[int, float] = {}
    for tok in toks:
        idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % DIM
        counts[idx] = counts.get(idx, 0.0) + 1.0
    for idx, c in counts.items():
        v[idx] = 1.0 + math.log(c)          # sublinear tf
    n = np.linalg.norm(v)
    return v / n if n else v


def embed(texts: list[str], provider: str | None = None,
          model: str | None = None) -> np.ndarray:
    provider = provider or config.EMBED_PROVIDER
    model = model or config.EMBED_MODEL
    if provider == "local":
        return np.vstack([_local_vector(t) for t in texts])

    out, missing, miss_idx = [None] * len(texts), [], []
    for i, t in enumerate(texts):
        key = hashlib.sha256(f"{model}\x00{t}".encode()).hexdigest()
        p = _CACHE / f"{key}.json"
        if p.exists():
            out[i] = np.array(json.loads(p.read_text()), dtype=np.float32)
            llm.SPEND.cached += 1
        else:
            missing.append(t)
            miss_idx.append(i)

    if missing:
        config.log(f"embeddings: {len(texts)} texts — {len(texts) - len(missing)} cached, "
                   f"{len(missing)} to fetch from {model}", indent=2)
    elif texts:
        config.log(f"embeddings: {len(texts)} texts, all cached", indent=2)

    batches = (len(missing) + 127) // 128
    for start in range(0, len(missing), 128):
        batch = missing[start:start + 128]
        config.log(f"batch {start // 128 + 1}/{batches} ({len(batch)} texts)", indent=3)
        resp = llm._openai_embed().embeddings.create(model=model, input=batch)
        llm.SPEND.add(model, resp.usage.total_tokens, 0)
        llm.SPEND.save()
        for j, item in enumerate(resp.data):
            i = miss_idx[start + j]
            vec = np.array(item.embedding, dtype=np.float32)
            vec /= (np.linalg.norm(vec) or 1.0)
            out[i] = vec
            key = hashlib.sha256(f"{model}\x00{texts[i]}".encode()).hexdigest()
            (_CACHE / f"{key}.json").write_text(json.dumps(vec.tolist()))
    return np.vstack(out)


def cosine(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return matrix @ query_vec
