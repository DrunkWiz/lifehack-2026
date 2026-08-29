"""
Thin wrapper around the OpenAI SDK.
Reads the API key from st.secrets["OPENAI_API_KEY"] (set via .streamlit/secrets.toml
locally, or the Streamlit Cloud secrets manager when deployed).

Adds two things borrowed from the bigger sibling build, folded in here rather
than a separate file:
- Disk cache, keyed by sha256 of everything that matters about the call. Re-running
  the app on the same product/prompt costs nothing after the first pass.
- A running spend ledger with a hard budget cap, so a runaway loop during testing
  can't burn an unbounded amount of API credit.
"""

import hashlib
import json
import pathlib
import streamlit as st
from openai import OpenAI

MODEL_TEXT = "gpt-4o-mini"     # cheap + fast, good enough for structured extraction/generation
MODEL_VISION = "gpt-4o-mini"   # gpt-4o-mini supports image input too
MODEL_EMBED = "text-embedding-3-small"   # ~$0.00002/product; 1536-dim
EMBED_BATCH = 100              # OpenAI accepts large batches; 100 keeps payloads modest

# USD per 1M tokens. Only used to estimate spend for the budget guard.
PRICING = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "text-embedding-3-small": {"in": 0.02, "out": 0.00},
}

CACHE_DIR = pathlib.Path(".cache/completions")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_PATH = pathlib.Path(".cache/spend.json")


class BudgetExceeded(RuntimeError):
    pass


def _default_budget() -> float:
    try:
        return float(st.secrets.get("AGENT_BUDGET_USD", 5.00))
    except Exception:
        return 5.00


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"calls": 0, "cached": 0, "usd": 0.0,
            "input_tokens": 0, "output_tokens": 0}


def _save_ledger(ledger: dict) -> None:
    try:
        LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    except OSError:
        pass


def _record_spend(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    ledger = _load_ledger()
    price = PRICING.get(model, {"in": 0.0, "out": 0.0})
    cost = prompt_tokens / 1e6 * price["in"] + completion_tokens / 1e6 * price["out"]
    ledger["calls"] = ledger.get("calls", 0) + 1
    ledger["usd"] = ledger.get("usd", 0.0) + cost
    ledger["input_tokens"] = ledger.get("input_tokens", 0) + prompt_tokens
    ledger["output_tokens"] = ledger.get("output_tokens", 0) + completion_tokens
    _save_ledger(ledger)


def _check_budget() -> None:
    ledger = _load_ledger()
    budget = _default_budget()
    if ledger.get("usd", 0.0) >= budget:
        raise BudgetExceeded(
            f"Spend ${ledger['usd']:.2f} has reached the ${budget:.2f} cap "
            f"(set AGENT_BUDGET_USD in secrets.toml to change it). "
            f"Delete .cache/spend.json to reset."
        )


def get_spend_summary() -> dict:
    summary = _load_ledger()
    summary["budget_usd"] = _default_budget()
    try:
        summary["api_key_configured"] = bool(st.secrets.get("OPENAI_API_KEY"))
    except Exception:
        summary["api_key_configured"] = False
    summary["text_model"] = MODEL_TEXT
    summary["embedding_model"] = MODEL_EMBED
    return summary


def _cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(key: str):
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            ledger = _load_ledger()
            ledger["cached"] = ledger.get("cached", 0) + 1
            _save_ledger(ledger)
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _cache_set(key: str, value) -> None:
    path = CACHE_DIR / f"{key}.json"
    try:
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


@st.cache_resource(show_spinner=False)
def get_client() -> OpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        st.error(
            "OPENAI_API_KEY not found in st.secrets.\n\n"
            "Locally: create `.streamlit/secrets.toml` with:\n"
            '`OPENAI_API_KEY = "sk-..."`\n\n'
            "On Streamlit Cloud: add it under App settings → Secrets."
        )
        st.stop()
    return OpenAI(api_key=api_key)


def _safe_json_parse(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: sometimes models wrap JSON in prose despite instructions.
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {"error": "parse_failed", "raw": raw}


def call_llm_json(system_prompt: str, user_prompt: str, model: str = MODEL_TEXT, temperature: float = 0.3) -> dict:
    key = _cache_key("json", model, str(temperature), system_prompt, user_prompt)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    _check_budget()
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    result = _safe_json_parse(resp.choices[0].message.content)
    if resp.usage:
        _record_spend(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    _cache_set(key, result)
    return result


def call_llm_vision_json(system_prompt: str, user_text: str, image_b64_list: list[str],
                          model: str = MODEL_VISION, temperature: float = 0.2) -> dict:
    key = _cache_key("vision", model, str(temperature), system_prompt, user_text, *image_b64_list)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    _check_budget()
    client = get_client()
    content = [{"type": "text", "text": user_text}]
    for b64 in image_b64_list:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )
    result = _safe_json_parse(resp.choices[0].message.content)
    if resp.usage:
        _record_spend(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    _cache_set(key, result)
    return result


def call_llm_text(system_prompt: str, user_prompt: str, model: str = MODEL_TEXT, temperature: float = 0.6) -> str:
    key = _cache_key("text", model, str(temperature), system_prompt, user_prompt)
    cached = _cache_get(key)
    if cached is not None:
        return cached["text"]

    _check_budget()
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = resp.choices[0].message.content
    if resp.usage:
        _record_spend(model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
    _cache_set(key, {"text": text})
    return text


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, batched. Used by retrieval.py to index the catalog twice
    (raw content vs agent-optimized content) and to embed the shopper's query."""
    client = get_client()
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        _check_budget()
        # the embeddings endpoint rejects empty strings, so blanks become a single space
        batch = [t if t and t.strip() else " " for t in texts[start:start + EMBED_BATCH]]
        resp = client.embeddings.create(model=MODEL_EMBED, input=batch)
        out.extend(item.embedding for item in resp.data)
        if resp.usage:
            _record_spend(MODEL_EMBED, resp.usage.prompt_tokens, 0)
    return out
