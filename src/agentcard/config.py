"""Paths, environment and category configs."""
from __future__ import annotations
import os, pathlib, functools, sys, time
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIXTURES = DATA / "fixtures"
CONFIGS = ROOT / "configs"
CACHE = ROOT / ".cache"
SCHEMA_PATH = ROOT / "agent_card_schema.json"

# Which catalogue variant this process operates on.
#   spec_rich — every row carries a custom spec metafield (weight, drop, pH,
#               comedogenic rating). Flattering to the raw baseline.
#   typical   — a plain Shopify export: title, body, tags, price, size.
# Set with AGENTCARD_VARIANT, or run both with `agentcard compare`.
VARIANT = os.environ.get("AGENTCARD_VARIANT", "spec_rich")
VARIANTS = ("spec_rich", "typical")

RAW = DATA / "raw" / VARIANT
if not RAW.exists() and (DATA / "raw").exists():
    RAW = DATA / "raw"                      # pre-variant layout, still works
OUT = ROOT / "out" if VARIANT == "spec_rich" else ROOT / "out" / VARIANT

for p in (OUT, CACHE):
    p.mkdir(parents=True, exist_ok=True)


_T0 = time.time()


def log(msg: str, indent: int = 1) -> None:
    """Progress to stderr, so stdout stays clean JSON you can pipe.

    Long stretches of silence during embedding read as a hang, and the first
    instinct is to Ctrl-C a run that is halfway through paying for itself.
    """
    print(f"{' ' * (indent * 2)}[{time.time() - _T0:6.1f}s] {msg}",
          file=sys.stderr, flush=True)


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


LLM_PROVIDER = env("AGENTCARD_LLM_PROVIDER", "openai")
EMBED_PROVIDER = env("AGENTCARD_EMBED_PROVIDER", "openai")
ENRICH_MODEL = env("AGENTCARD_ENRICH_MODEL", "gpt-4.1")
CHEAP_MODEL = env("AGENTCARD_CHEAP_MODEL", "gpt-4.1-mini")
EMBED_MODEL = env("AGENTCARD_EMBED_MODEL", "text-embedding-3-small")
# Reading a catalogue page that has no selectable text — a lookbook, a scanned
# brochure. Cheap model is fine; it is transcription, not reasoning.
VISION_MODEL = env("AGENTCARD_VISION_MODEL", "gpt-4o-mini")
BUDGET_USD = float(env("AGENTCARD_BUDGET_USD", "5.00"))

# Point the pipeline at any OpenAI-compatible endpoint — a self-hosted model, a
# vendor's own commerce LLM, an Azure deployment — without touching code. The
# enricher only needs chat completions with structured outputs; the retriever
# only needs an embeddings endpoint. Nothing here is specific to OpenAI beyond
# the wire format.
LLM_BASE_URL = env("AGENTCARD_LLM_BASE_URL") or None
EMBED_BASE_URL = env("AGENTCARD_EMBED_BASE_URL") or None

# USD per 1M tokens. Update if pricing moves; only used for the budget guard.
PRICING = {
    "gpt-4.1":       {"in": 2.00, "out": 8.00},
    "gpt-4.1-mini":  {"in": 0.40, "out": 1.60},
    "gpt-4o":        {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":   {"in": 0.15, "out": 0.60},
    "text-embedding-3-small": {"in": 0.02, "out": 0.0},
    "text-embedding-3-large": {"in": 0.13, "out": 0.0},
    "gpt-4.1-nano":  {"in": 0.10, "out": 0.40},
}

# Which catalogue file belongs to which category config.
CATALOGUES = {
    "footwear.running": {"csv": RAW / "shopify_running_shoes.csv",
                         "config": CONFIGS / "footwear_running.yaml"},
    "skincare.facial":  {"csv": RAW / "shopify_facial_skincare.csv",
                         "config": CONFIGS / "skincare_facial.yaml"},
}


@functools.lru_cache(maxsize=None)
def load_category_config(path_or_category: str) -> dict:
    """Accepts a category id ('footwear.running') or a path to a YAML file."""
    if path_or_category in CATALOGUES:
        path = CATALOGUES[path_or_category]["config"]
    else:
        path = pathlib.Path(path_or_category)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def all_category_configs() -> dict[str, dict]:
    return {cat: load_category_config(cat) for cat in CATALOGUES}
