"""Agent Card schema: loading, OpenAI strict-mode conversion, validation, accessors.

The strict-mode conversion is the piece that usually eats an evening. OpenAI
structured outputs reject a draft-07 schema unless every object sets
additionalProperties:false and lists every property in `required`, and it
rejects several keywords outright. `to_strict()` does that transform so the
authored schema in the repo root stays the readable source of truth.
"""
from __future__ import annotations
import copy, json, functools
from jsonschema import Draft7Validator
from .config import SCHEMA_PATH

# Keywords the structured-outputs endpoint does not accept.
_STRIP = {"minLength", "maxLength", "pattern", "format", "minimum", "maximum",
          "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "default",
          "minItems", "maxItems", "uniqueItems", "$schema"}


@functools.lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def to_strict(schema: dict | None = None) -> dict:
    """Return a copy that satisfies OpenAI structured-output strict mode."""
    node = copy.deepcopy(schema if schema is not None else load_schema())

    def walk(n):
        if isinstance(n, list):
            for i in n:
                walk(i)
            return
        if not isinstance(n, dict):
            return
        for k in list(n):
            if k in _STRIP:
                n.pop(k)
        if n.get("type") == "object" or "properties" in n:
            props = n.setdefault("properties", {})
            n["additionalProperties"] = False
            # strict mode: every property must be required
            n["required"] = list(props.keys())
            for v in props.values():
                walk(v)
        if "items" in n:
            walk(n["items"])

    walk(node)
    # category_extension is intentionally open-ended; strict mode cannot express
    # that, so it becomes a JSON string the caller parses.
    ce = node.get("properties", {}).get("category_extension")
    if ce is not None:
        node["properties"]["category_extension"] = {
            "type": "string",
            "description": "JSON object encoded as a string: category-specific "
                           "attributes driven by the category config.",
        }
    node["additionalProperties"] = False
    node["required"] = list(node.get("properties", {}).keys())
    return node


def response_format(name: str = "agent_card") -> dict:
    return {"type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": to_strict()}}


def validate(card: dict) -> list[str]:
    """Return a list of human-readable validation errors ([] means valid)."""
    v = Draft7Validator(load_schema())
    return [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in sorted(v.iter_errors(card), key=lambda e: list(e.path))]


# ------------------------------------------------------------- accessors ---
def numeric(card: dict, key: str):
    for n in card.get("hard_constraints", {}).get("numeric", []) or []:
        if n.get("key") == key:
            return n.get("value")
    return None


def categorical(card: dict, key: str) -> list[str]:
    for c in card.get("hard_constraints", {}).get("categorical", []) or []:
        if c.get("key") == key:
            return [str(v) for v in c.get("values", [])]
    return []


def has_constraint(card: dict, key: str) -> bool:
    return numeric(card, key) is not None or bool(categorical(card, key))


def situational_tags(card: dict) -> list[str]:
    return list(card.get("hard_constraints", {}).get("situational_tags", []) or [])
