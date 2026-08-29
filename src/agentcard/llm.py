"""Provider abstraction + disk cache + budget guard.

Two providers:
  openai — real API calls, structured outputs, cached to disk by sha256(prompt)
  local  — deterministic offline stand-in, no network, no cost

Every completion is cached before it is returned. Re-running the pipeline after
a crash costs nothing, which is the difference between a $3 hackathon and a $40
one.
"""
from __future__ import annotations
import hashlib, json, os, time, pathlib
from dataclasses import dataclass, field
from . import config

CACHE_DIR = config.CACHE / "completions"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LEDGER = config.CACHE / "spend.json"


def _key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class Spend:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, pin: int, pout: int) -> None:
        price = config.PRICING.get(model, {"in": 0.0, "out": 0.0})
        cost = pin / 1e6 * price["in"] + pout / 1e6 * price["out"]
        self.calls += 1
        self.prompt_tokens += pin
        self.completion_tokens += pout
        self.usd += cost
        m = self.by_model.setdefault(model, {"calls": 0, "usd": 0.0})
        m["calls"] += 1
        m["usd"] += cost

    def save(self) -> None:
        LEDGER.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


SPEND = Spend()
if LEDGER.exists():
    try:
        SPEND.__dict__.update(json.loads(LEDGER.read_text(encoding="utf-8")))
    except Exception:
        pass


class BudgetExceeded(RuntimeError):
    pass


def _check_budget() -> None:
    if SPEND.usd >= config.BUDGET_USD:
        raise BudgetExceeded(
            f"spend ${SPEND.usd:.2f} has reached the ${config.BUDGET_USD:.2f} cap "
            f"(AGENTCARD_BUDGET_USD). Clear .cache/spend.json to reset."
        )


# ------------------------------------------------------------- openai ------
_client = None
_embed_client = None


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI
        kw = {"base_url": config.LLM_BASE_URL} if config.LLM_BASE_URL else {}
        _client = OpenAI(**kw)
    return _client


def _openai_embed():
    """Embeddings may live behind a different endpoint from the chat model."""
    global _embed_client
    if config.EMBED_BASE_URL is None:
        return _openai()
    if _embed_client is None:
        from openai import OpenAI
        _embed_client = OpenAI(base_url=config.EMBED_BASE_URL)
    return _embed_client


def complete_json(system: str, user: str, *, schema_format: dict,
                  model: str | None = None, provider: str | None = None) -> dict:
    """Structured-output completion. Cached by sha256 of everything that matters."""
    provider = provider or config.LLM_PROVIDER
    model = model or config.ENRICH_MODEL
    ck = _key(provider, model, system, user, json.dumps(schema_format, sort_keys=True))
    path = CACHE_DIR / f"{ck}.json"
    if path.exists():
        SPEND.cached += 1
        return json.loads(path.read_text(encoding="utf-8"))["content"]

    if provider == "local":
        from .local_provider import local_agent_card
        content = local_agent_card(user)
    else:
        _check_budget()
        resp = _openai().chat.completions.create(
            model=model, temperature=0.2, response_format=schema_format,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        content = json.loads(resp.choices[0].message.content)
        u = resp.usage
        SPEND.add(model, u.prompt_tokens, u.completion_tokens)
        SPEND.save()

    path.write_text(json.dumps(
        {"content": content, "model": model, "provider": provider,
         "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        indent=2, ensure_ascii=False), encoding="utf-8")
    return content


def complete_text(system: str, user: str, *, model: str | None = None,
                  temperature: float = 0.9, provider: str | None = None) -> str:
    provider = provider or config.LLM_PROVIDER
    model = model or config.CHEAP_MODEL
    ck = _key("text", provider, model, system, user, str(temperature))
    path = CACHE_DIR / f"{ck}.json"
    if path.exists():
        SPEND.cached += 1
        return json.loads(path.read_text(encoding="utf-8"))["content"]

    if provider == "local":
        from .local_provider import local_queries
        content = local_queries(user)
    else:
        _check_budget()
        resp = _openai().chat.completions.create(
            model=model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content
        u = resp.usage
        SPEND.add(model, u.prompt_tokens, u.completion_tokens)
        SPEND.save()

    path.write_text(json.dumps({"content": content, "model": model,
                                "provider": provider}, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return content


def complete_vision_json(system: str, user: str, images_b64: list[str],
                         model: str | None = None,
                         provider: str | None = None) -> dict:
    """Read a page that has no usable text.

    Cached on the image bytes, so re-running an extraction over a 40-page
    lookbook costs nothing the second time — which matters, because vision
    calls are the most expensive thing this pipeline does.
    """
    provider = provider or config.LLM_PROVIDER
    model = model or config.VISION_MODEL
    ck = _key("vision", provider, model, system, user, *images_b64)
    path = CACHE_DIR / f"{ck}.json"
    if path.exists():
        SPEND.cached += 1
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    if provider == "local":
        raise RuntimeError("vision extraction needs a model provider")

    _check_budget()
    content = [{"type": "text", "text": user}]
    for b64 in images_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}})
    resp = _openai().chat.completions.create(
        model=model, temperature=0.2, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": content}])
    try:
        out = json.loads(resp.choices[0].message.content)
    except json.JSONDecodeError:
        raw = resp.choices[0].message.content or ""
        a, b = raw.find("{"), raw.rfind("}")
        out = json.loads(raw[a:b + 1]) if a != -1 and b > a else {"products": []}
    u = resp.usage
    SPEND.add(model, u.prompt_tokens, u.completion_tokens)
    SPEND.save()
    path.write_text(json.dumps({"content": out, "model": model, "provider": provider},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def probe_openai() -> tuple[bool, str]:
    """Cheap reachability check so the runner can fall back cleanly."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False, "OPENAI_API_KEY is not set"
    try:
        _openai().models.list()
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def available_models() -> list[str]:
    try:
        return sorted(m.id for m in _openai().models.list().data)
    except Exception:  # noqa: BLE001
        return []


def check_models() -> dict:
    """Which configured models does this key actually have access to?

    Model availability varies by account and by project key. Finding out here
    costs one list call; finding out during enrichment costs a failed run.
    """
    have = set(available_models())
    if not have:
        return {}
    out = {}
    for role, name in (("enrich", config.ENRICH_MODEL),
                       ("cheap", config.CHEAP_MODEL),
                       ("embed", config.EMBED_MODEL)):
        if name in have:
            out[role] = {"model": name, "available": True}
        else:
            family = name.split("-")[0]
            near = [m for m in sorted(have) if m.startswith(family)][:6]
            out[role] = {"model": name, "available": False, "try_instead": near}
    return out
