"""Guardrails for the things that break at a booth."""
import json, pathlib, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentcard import config, ingest, readiness, retrieve, schema  # noqa: E402

OUT = config.OUT
pytestmark = pytest.mark.skipif(
    not (OUT / "agent_cards.json").exists(),
    reason="run `python -m agentcard all` first")


@pytest.fixture(scope="module")
def cards():
    return json.loads((OUT / "agent_cards.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def configs():
    return config.all_category_configs()


def test_ingest_reads_both_catalogues():
    rows = ingest.load_all()
    assert len(rows) >= 40
    assert {r["category"] for r in rows} == set(config.CATALOGUES)


def test_strict_schema_is_openai_compatible():
    strict = schema.to_strict()

    def walk(n, path="root"):
        if isinstance(n, dict):
            if n.get("type") == "object" or "properties" in n:
                assert n.get("additionalProperties") is False, path
                assert set(n.get("required", [])) == set(n.get("properties", {})), path
            for k, v in n.items():
                if k not in ("required",):
                    walk(v, f"{path}.{k}")
        elif isinstance(n, list):
            for i, v in enumerate(n):
                walk(v, f"{path}[{i}]")
    walk(strict)


def test_every_card_validates(cards):
    bad = {k: schema.validate(v) for k, v in cards.items()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, json.dumps(bad, indent=2)[:2000]


def test_negative_information_is_mandatory(cards):
    """Rule 3 of the enrichment prompt. The whole point of the card."""
    thin = [k for k, c in cards.items() if len(c.get("not_for", [])) < 2]
    assert not thin, f"cards with fewer than two not_for entries: {thin}"


def test_use_cases_cite_constraints(cards):
    for pid, c in cards.items():
        for u in c["use_cases"]:
            assert u.get("grounded_in"), f"{pid}: use case with no grounded_in"


def test_readiness_is_deterministic(cards, configs):
    pid = next(iter(cards))
    cfg = configs[cards[pid]["identity"]["category"]]
    a = readiness.readiness(cards[pid], cfg)
    b = readiness.readiness(cards[pid], cfg)
    assert a == b
    assert 0 <= a[0] <= 100


def test_enrichment_beats_raw_on_readiness(cards, configs):
    base = json.loads((OUT / "raw_baseline_cards.json").read_text(encoding="utf-8"))
    worse = [p for p in cards
             if cards[p]["readiness"]["score"] <= base[p]["readiness"]["score"]]
    assert not worse, f"enrichment did not improve: {worse}"


def test_all_positive_persona_cards_are_penalised(configs):
    cfg = next(iter(configs.values()))
    card = {"hard_constraints": {"numeric": [], "categorical": []},
            "use_cases": [], "not_for": [], "comparisons": [],
            "personas": [{"label": "everyone", "fit": "strong", "reasoning": "x"}],
            "provenance": {"field_sources": [], "unsupported_claims": []}}
    _, parts, gaps = readiness.readiness(card, cfg)
    assert parts["persona_coverage"] < 0.25
    assert any("poor" in g for g in gaps)


@pytest.mark.parametrize("query,expect_key,expect_val", [
    ("lightweight trail shoes under S$200", "surface", "trail"),
    ("I have wide feet and need a daily trainer", "width", "wide"),
    ("fragrance-free moisturiser for sensitive skin", "fragrance_free", "true"),
])
def test_query_parser_extracts_hard_constraints(query, expect_key, expect_val, configs):
    f = retrieve.parse_query(query, configs)
    assert f.categorical.get(expect_key) == expect_val, f.describe()


def test_budget_ceiling_is_respected(configs):
    f = retrieve.parse_query("running shoes under S$180", configs)
    assert f.price_max == 180
    ids = retrieve.sql_filter(f)
    import sqlite3
    con = sqlite3.connect(retrieve.indexer.DB)
    prices = [r[0] for r in con.execute(
        "SELECT price FROM products WHERE id IN (%s)" % ",".join("?" * len(ids)), ids)]
    assert all(p <= 180 for p in prices)


def test_out_of_stock_is_never_recommended(cards):
    res = retrieve.search("a good daily running shoe", k=5)
    import sqlite3
    con = sqlite3.connect(retrieve.indexer.DB)
    for h in res["hits"]:
        av = con.execute("SELECT availability FROM products WHERE id=?", (h["id"],)).fetchone()[0]
        assert av != "out_of_stock"


def test_enrichment_path_never_reads_the_ground_truth_fixture():
    """The benchmark is only honest if the enricher works from the catalogue alone.

    `data/fixtures/ground_truth.json` drives query generation and the offline
    provider. If any of it reached the enrichment prompt, the enriched arm would
    be scoring against answers it had been handed.

    Checked structurally rather than by searching the prompt for values: a heel
    drop of 8mm is the string "8", which also appears in a price. The property
    that actually matters is that no module on the enrichment path touches the
    fixture at all.
    """
    src_dir = ROOT / "src" / "agentcard"
    on_path = ["ingest.py", "prompts.py", "enrich.py", "schema.py", "readiness.py"]
    for name in on_path:
        text = (src_dir / name).read_text(encoding="utf-8")
        assert "ground_truth" not in text, (
            f"{name} is on the enrichment path and references the ground-truth "
            f"fixture — the enriched arm would be scoring against its own answers")

    # And the fixture is genuinely used elsewhere, so this test is not vacuous.
    users = [f.name for f in src_dir.glob("*.py")
             if "ground_truth" in f.read_text(encoding="utf-8")]
    assert set(users) <= {"local_provider.py", "simulate.py"}, users
    assert users, "nothing reads the fixture — is the test looking at the right thing?"


def test_config_can_be_inferred_from_an_unseen_catalogue(configs):
    """Generalisability, checked rather than asserted.

    A hand-written config per category is fine for a demo and useless for a
    brand uploading something nobody has seen. Inference has to produce a
    config the rest of the pipeline can actually load.
    """
    from agentcard import infer_config
    rows = [r for r in ingest.load_all() if r["category"] == "footwear.running"]
    cfg = infer_config.infer(rows, "Running shoes", "footwear.running",
                             provider="local")
    for key in infer_config.REQUIRED_KEYS:
        assert key in cfg, key
    assert cfg["required_numeric"], "no numeric attributes recovered"
    assert cfg["category_cues"], "no cues — queries could not be routed here"
    # It must not invent a hard filter it cannot justify: eliminating on a
    # guessed constraint silently removes correct answers.
    hard = set(cfg["hard_filter_keys"])
    known = set(cfg["required_categorical"]) | set(cfg["constraint_synonyms"])
    assert hard <= known, f"hard filter on unknown keys: {hard - known}"
    # And it must round-trip through YAML into something the retriever accepts.
    import yaml
    reloaded = yaml.safe_load(infer_config.to_yaml(cfg))
    f = retrieve.parse_query("lightweight trail shoes under $200",
                             {"footwear.running": reloaded})
    assert f.category == "footwear.running"


def test_inferred_vocabulary_excludes_brand_names(configs):
    """A situational vocabulary full of vendors routes every query to whoever
    sells the most SKUs. Shoppers describe situations, not brands."""
    from agentcard import infer_config
    rows = ingest.load_all()
    brands = {b.lower() for r in rows for b in r["brand"].split()}
    cfg = infer_config.infer(rows, "Everything", "test.all", provider="local")
    leaked = [t for t in cfg["situational_vocabulary"] + cfg["category_cues"]
              if t.replace("_", " ") in brands]
    assert not leaked, f"brand names leaked into the vocabulary: {leaked}"


def test_significance_is_reported():
    p = OUT / "simulator_report.json"
    if not p.exists():
        pytest.skip("no simulator report")
    rep = json.loads(p.read_text(encoding="utf-8"))
    assert "significance" in rep, "the report must state whether the lift is real"
    for pair, v in rep["significance"].items():
        assert "p_value" in v and "discordant" in v, pair


def test_simulator_shows_a_real_lift():
    p = OUT / "simulator_report.json"
    if not p.exists():
        pytest.skip("no simulator report")
    rep = json.loads(p.read_text(encoding="utf-8"))
    r = rep["recall_at_k"]
    assert r["enriched"] > r["raw"], "enrichment must beat the raw catalogue"
    assert r["enriched+sql"] >= r["enriched"] - 1e-9, "the hard filter must not hurt"
