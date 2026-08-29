"""
Thin wrapper around the OpenAI SDK.
Reads the API key from st.secrets["OPENAI_API_KEY"] (set via .streamlit/secrets.toml
locally, or the Streamlit Cloud secrets manager when deployed).
"""

import json
import streamlit as st
from openai import OpenAI

MODEL_TEXT = "gpt-4o-mini"     # cheap + fast, good enough for structured extraction/generation
MODEL_VISION = "gpt-4o-mini"   # gpt-4o-mini supports image input too
MODEL_EMBED = "text-embedding-3-small"   # ~$0.00002/product; 1536-dim
EMBED_BATCH = 100              # OpenAI accepts large batches; 100 keeps payloads modest


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
    return _safe_json_parse(resp.choices[0].message.content)


def call_llm_vision_json(system_prompt: str, user_text: str, image_b64_list: list[str],
                          model: str = MODEL_VISION, temperature: float = 0.2) -> dict:
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
    return _safe_json_parse(resp.choices[0].message.content)


def call_llm_text(system_prompt: str, user_prompt: str, model: str = MODEL_TEXT, temperature: float = 0.6) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings, batched. Used by retrieval.py to index the catalog twice
    (raw content vs agent-optimized content) and to embed the shopper's query."""
    client = get_client()
    out: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        # the embeddings endpoint rejects empty strings, so blanks become a single space
        batch = [t if t and t.strip() else " " for t in texts[start:start + EMBED_BATCH]]
        resp = client.embeddings.create(model=MODEL_EMBED, input=batch)
        out.extend(item.embedding for item in resp.data)
    return out
