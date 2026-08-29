"""Clustering, config inference from clusters, and the card-to-copy surfaces."""
import json, pathlib, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentcard import cluster, config, copy_out, infer_config, ingest, schema  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return ingest.load_all()


def test_clustering_loses_no_products(rows):
    """A product that falls out of every cluster never gets enriched, and the
    loss is invisible unless something checks for it."""
    cl = cluster.cluster(rows, provider="local")
    covered = [i for c in cl for i in c["product_indices"]]
    assert sorted(covered) == list(range(len(rows)))
    assert len(covered) == len(set(covered)), "a product landed in two clusters"


def test_a_cluster_yields_a_usable_config(rows):
    cl = cluster.cluster(rows, provider="local")
    biggest = max(cl, key=lambda c: len(c["product_indices"]))
    sub = [rows[i] for i in biggest["product_indices"]]
    cfg = infer_config.infer(sub, biggest["cluster_name"],
                             "auto." + cluster.slug(biggest["cluster_name"]),
                             provider="local")
    for key in infer_config.REQUIRED_KEYS:
        assert key in cfg


def test_expected_attributes_widen_the_required_set(rows):
    """Column statistics can only find what is present. A readiness score has to
    know about attributes the catalogue is missing entirely."""
    sub = [r for r in rows if r["category"] == "footwear.running"]
    plain = infer_config.infer(sub, "Running shoes", "x.y", provider="local")
    widened = infer_config.infer(sub, "Running shoes", "x.y", provider="local",
                                 expected=["warranty_months", "outsole_material"])
    added = (set(widened["required_numeric"]) | set(widened["required_categorical"])) - \
            (set(plain["required_numeric"]) | set(plain["required_categorical"]))
    assert {"warranty_months", "outsole_material"} <= added
    assert "warranty_months" in widened["required_numeric"], "a _months key is numeric"


@pytest.fixture(scope="module")
def card():
    p = config.OUT / "agent_cards.json"
    if not p.exists():
        pytest.skip("run `python -m agentcard enrich` first")
    return next(iter(json.loads(p.read_text(encoding="utf-8")).values()))


def test_passage_invents_nothing(card):
    """Every sentence must be assembled from the card, never generated."""
    text = copy_out.passage(card)
    assert card["identity"]["title"] in text
    for n in card["not_for"][:1]:
        assert n["reason"].rstrip(".") in text
    assert "Who should skip this?" in text


def test_passage_does_not_repeat_the_specifications(card):
    text = copy_out.passage(card)
    assert text.count("Specifications:") <= 1


def test_json_ld_is_valid_product_markup(card):
    d = copy_out.json_ld(card)
    assert d["@context"] == "https://schema.org" and d["@type"] == "Product"
    assert d["offers"]["priceCurrency"] and d["offers"]["price"] is not None
    assert d["offers"]["availability"].startswith("https://schema.org/")
    names = {p["name"] for p in d["additionalProperty"]}
    for n in card["hard_constraints"]["numeric"]:
        assert n["key"] in names
    json.dumps(d)          # must be serialisable as-is for a <script> tag


def test_json_ld_carries_the_negative_information(card):
    d = copy_out.json_ld(card)
    faq = d.get("subjectOf", {}).get("mainEntity", [])
    assert faq, "no FAQ block — the exclusions never reach the page"
    answers = " ".join(q["acceptedAnswer"]["text"] for q in faq)
    assert any(n["reason"].rstrip(".") in answers for n in card["not_for"])


def test_review_notes_surface_inferred_claims(card):
    notes = copy_out.review_notes(card)
    inferred = [f for f in card["provenance"]["field_sources"]
                if f.get("source") == "inferred"]
    if inferred:
        assert notes, "inferred fields exist but nothing was flagged for review"
        assert any("inferred" in n for n in notes)


def test_exclusions_are_not_inverted():
    """A tag in not_for means the product is wrong for that situation. Reusing
    the tag's own exclusion text inverts it — a barrier balm came out as 'not
    for dry skin', which is the opposite of true."""
    cards = json.loads((config.OUT / "agent_cards.json").read_text(encoding="utf-8"))
    balm = cards.get("sable-barrier-balm")
    if not balm:
        pytest.skip("demo catalogue not enriched")
    excl = " ".join(n["exclusion"].lower() for n in balm["not_for"])
    tags = set(schema.situational_tags(balm))
    assert "sensitive_barrier_repair" in tags
    assert "dry" not in excl, f"balm for dry, sensitive skin excludes dry skin: {excl}"
