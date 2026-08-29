"""Exercise the real-provider plumbing without a network call.

The offline provider proves the pipeline; it does not prove that the OpenAI
path is wired correctly. These tests stub the client so the first live run
fails for interesting reasons, not for plumbing ones.
"""
import json, pathlib, sys, types
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentcard import config, enrich, ingest, llm, prompts, schema  # noqa: E402


def _fake_card(row_id="kestrel-drift-3"):
    """Shaped exactly like a strict-mode structured output: category_extension
    arrives as a JSON *string*, not an object."""
    return {
        "id": row_id,
        "identity": {"title": "Kestrel Drift 3", "brand": "Kestrel",
                     "category": "footwear.running",
                     "price": {"amount": 179.0, "currency": "SGD", "band": "mid"},
                     "availability": "in_stock"},
        "hard_constraints": {
            "numeric": [{"key": "weight_g", "value": 258, "unit": "g"},
                        {"key": "heel_drop_mm", "value": 8, "unit": "mm"},
                        {"key": "stack_height_mm", "value": 34, "unit": "mm"}],
            "categorical": [{"key": "surface", "values": ["road"]},
                            {"key": "arch_support", "values": ["neutral"]},
                            {"key": "width", "values": ["standard"]},
                            {"key": "closure", "values": ["lace"]}],
            "situational_tags": ["humid_climate", "daily_training"]},
        "use_cases": [{"scenario": "Daily easy mileage in humidity",
                       "why_it_fits": "258g with an engineered mesh upper",
                       "grounded_in": ["weight_g"], "confidence": 0.9}],
        "personas": [{"label": "Beginner runner", "fit": "strong",
                      "reasoning": "forgiving ride", "experience_level": "beginner"},
                     {"label": "Runner with wide feet", "fit": "poor",
                      "reasoning": "standard last", "experience_level": "any"}],
        "not_for": [{"exclusion": "Runners with wide feet (2E+)",
                     "reason": "standard last", "source": "spec"},
                    {"exclusion": "Trail running", "reason": "road outsole",
                     "source": "spec"}],
        "comparisons": [{"against": "Vela Aero 9", "axis": "weight", "direction": "more",
                         "magnitude": "22g", "tradeoff": "more cushioning, less turnover"}],
        "narrative": {"one_line_pitch": "An honest daily trainer.",
                      "intent_variants": [{"intent": "humid_climate", "copy": "Vents well."}]},
        "category_extension": json.dumps({"handle": row_id, "archetype": "daily"}),
        "provenance": {"field_sources": [{"field_path": "hard_constraints.numeric.weight_g",
                                          "source": "catalog_spec", "evidence": "Weight: 258g"}],
                       "unsupported_claims": [],
                       "enriched_at": "2026-08-29T00:00:00Z", "model": "gpt-4.1"},
    }


class _FakeClient:
    """Records what the pipeline actually sends to the API."""
    def __init__(self, payload):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                msg = types.SimpleNamespace(content=json.dumps(payload))
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=msg)],
                    usage=types.SimpleNamespace(prompt_tokens=1200, completion_tokens=800))
        self.chat = types.SimpleNamespace(completions=_Completions())


@pytest.fixture
def stubbed(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "completions")
    llm.CACHE_DIR.mkdir(parents=True)
    monkeypatch.setattr(llm, "LEDGER", tmp_path / "spend.json")
    monkeypatch.setattr(llm, "SPEND", llm.Spend())
    client = _FakeClient(_fake_card())
    monkeypatch.setattr(llm, "_openai", lambda: client)
    return client


def test_request_shape_is_what_the_api_expects(stubbed):
    llm.complete_json("sys", "user", schema_format=schema.response_format(),
                      model="gpt-4.1", provider="openai")
    kw = stubbed.calls[0]
    assert kw["model"] == "gpt-4.1"
    rf = kw["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["additionalProperties"] is False
    assert [m["role"] for m in kw["messages"]] == ["system", "user"]


def test_second_identical_call_is_free(stubbed):
    for _ in range(2):
        llm.complete_json("sys", "user", schema_format=schema.response_format(),
                          model="gpt-4.1", provider="openai")
    assert len(stubbed.calls) == 1, "cache did not prevent the second call"
    assert llm.SPEND.calls == 1 and llm.SPEND.cached == 1
    assert llm.SPEND.usd > 0


def test_budget_ceiling_blocks_the_next_call(stubbed, monkeypatch):
    monkeypatch.setattr(config, "BUDGET_USD", 0.0001)
    llm.complete_json("sys", "a", schema_format=schema.response_format(),
                      model="gpt-4.1", provider="openai")
    with pytest.raises(llm.BudgetExceeded):
        llm.complete_json("sys", "b", schema_format=schema.response_format(),
                          model="gpt-4.1", provider="openai")


def test_strict_mode_output_round_trips_to_a_valid_card(stubbed):
    card = enrich.postprocess(_fake_card())
    assert isinstance(card["category_extension"], dict), \
        "category_extension must be parsed back from its strict-mode string form"
    assert schema.validate(card) == []


def test_full_row_enrichment_against_the_stub(stubbed):
    rows = ingest.load_all()
    row = next(r for r in rows if r["id"] == "kestrel-drift-3")
    cfg = config.load_category_config("footwear.running")
    card = enrich.enrich_row(row, rows, cfg, ingest.load_reviews(), provider="openai")
    assert schema.validate(card) == []
    assert card["readiness"]["score"] > 0
    user_msg = stubbed.calls[0]["messages"][1]["content"]
    assert "CATEGORY CONFIG:" in user_msg and "RAW CATALOG ROW:" in user_msg
    assert "COMPETITOR CONTEXT" in user_msg


def test_system_prompt_matches_the_spec_document():
    """If the spec is edited, the prompt must be edited with it."""
    spec = (ROOT / "enrichment_prompt_and_scoring.md").read_text(encoding="utf-8")
    for rule in ("NEGATIVE INFORMATION IS MANDATORY", "SITUATIONAL TAGS bridge",
                 "USE CASES MUST CITE SPECS", "PROVENANCE"):
        assert rule in prompts.SYSTEM and rule in spec
