"""Booth demo.

Three tabs, in the order a judge asks about them:
  Ask        live intent query, side by side against what today's catalogue does
  Before/After   one product, raw row in, Agent Card out, readiness delta
  Evidence   the simulator numbers, and how they were produced

Reads pre-generated JSON. No API call is made to answer a query unless the
embedding provider is set to openai, so the demo survives conference wifi.
"""
from __future__ import annotations
import json, pathlib, sys
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from agentcard import (catalog_loader, cluster, config, copy_out,  # noqa: E402
                       embed as embedder, enrich, index as indexer,
                       infer_config, ingest, readiness, retrieve, schema)

st.set_page_config(page_title="Agent Card — AI-ready product content",
                   page_icon="🛍️", layout="wide")

CSS = """
<style>
  .stApp { background: #0e1117; }
  .card { border:1px solid #2a2f3a; border-radius:10px; padding:14px 16px; margin-bottom:10px;
          background:#151922; }
  .muted { color:#8b93a7; font-size:0.85rem; }
  .pill { display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.72rem;
          background:#1e2530; color:#9fb4d8; margin-right:6px; border:1px solid #2a3444;}
  .pill-bad { background:#2a1a1a; color:#e08585; border-color:#4a2a2a; }
  .pill-good{ background:#132a1e; color:#6fce9a; border-color:#1e4a33; }
  .big { font-size:2.1rem; font-weight:650; line-height:1.1; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_all():
    cards = json.loads((config.OUT / "agent_cards.json").read_text(encoding="utf-8"))
    base = json.loads((config.OUT / "raw_baseline_cards.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in ingest.load_all()}
    rep = {}
    p = config.OUT / "simulator_report.json"
    if p.exists():
        rep = json.loads(p.read_text(encoding="utf-8"))
    cmp_ = {}
    p = config.ROOT / "out" / "variant_comparison.json"
    if p.exists():
        cmp_ = json.loads(p.read_text(encoding="utf-8"))
    val = {}
    p = config.OUT / "score_validation.json"
    if p.exists():
        val = json.loads(p.read_text(encoding="utf-8"))
    return cards, base, rows, rep, cmp_, val


try:
    CARDS, BASE, ROWS, REPORT, COMPARISON, VALIDATION = load_all()
except FileNotFoundError:
    st.error("No Agent Cards yet. Run `python -m agentcard all` first.")
    st.stop()

CONFIGS = config.all_category_configs()

with st.sidebar:
    st.markdown("### Agent Card")
    st.caption("Answer Engine Optimisation for product catalogues — a structured "
               "knowledge layer that lets an AI assistant reason about, and "
               "confidently recommend, a brand's products.")
    st.metric("Products enriched", len(CARDS))
    st.metric("Categories", len(CONFIGS))
    if REPORT:
        r = REPORT["recall_at_k"]
        st.metric(f"Recall@{REPORT['k']} — enriched + SQL", f"{r['enriched+sql']}%",
                  delta=f"{REPORT['lift_points']:+} pts vs raw catalogue")
    st.divider()
    st.caption(f"embeddings: `{config.EMBED_PROVIDER}` · enrichment: `{config.LLM_PROVIDER}`")
    st.caption("Adding a category is one YAML file in `configs/`. "
               "Nothing in the retriever knows what a running shoe is.")

tab_up, tab_ask, tab_cmp, tab_ba, tab_ev = st.tabs(
    ["Onboard", "Ask", "Compare", "Before / After", "Evidence"])


# ---------------------------------------------------------------- Ask ------
def render_hit(pid: str, score: float, rank: int, show_card: bool = True):
    c = CARDS[pid]
    idy = c["identity"]
    read = c.get("readiness", {})
    base_score = BASE.get(pid, {}).get("readiness", {}).get("score", 0)
    st.markdown(
        f"<div class='card'><b>{rank}. {idy['title']}</b> &nbsp; "
        f"<span class='muted'>SGD {idy['price']['amount']:.0f} · {idy.get('availability','')}"
        f" · match {score:.3f}</span><br>"
        f"<span class='muted'>{c.get('narrative',{}).get('one_line_pitch','')}</span><br>"
        f"<span class='pill pill-good'>readiness {read.get('score',0):.0f}</span>"
        f"<span class='pill'>was {base_score:.0f}</span>"
        + "".join(f"<span class='pill'>{t}</span>"
                  for t in c['hard_constraints'].get('situational_tags', [])[:4])
        + "</div>", unsafe_allow_html=True)
    if not show_card:
        return
    with st.expander("Why this was surfaced"):
        for u in c["use_cases"][:3]:
            st.markdown(f"**{u['scenario']}** — {u['why_it_fits']}  \n"
                        f"<span class='muted'>grounded in: "
                        f"{', '.join(u.get('grounded_in', []))} · confidence {u['confidence']}</span>",
                        unsafe_allow_html=True)
        st.markdown("**Not for**")
        for n in c["not_for"][:3]:
            st.markdown(f"- {n['exclusion']} — {n['reason']} `{n.get('source','')}`")
        if c.get("comparisons"):
            st.markdown("**Against siblings**")
            for cm in c["comparisons"][:2]:
                st.markdown(f"- {cm['direction']} {cm['axis']} than {cm['against']} "
                            f"({cm.get('magnitude','')}) — tradeoff: {cm.get('tradeoff','')}")


with tab_ask:
    st.markdown("#### Ask the way a shopper actually asks")
    examples = [
        "I'm training for my first half marathon in Singapore's humid weather, budget under S$200",
        "I have flat feet and my ankles roll in on long runs. What should I get?",
        "Something for oily skin in this humidity that takes under 5 minutes in the morning",
        "Fragrance-free moisturiser for a barrier that's stinging after a peel",
        "Wide feet, heavier runner, need a daily trainer that won't bottom out",
        "My hands hurt from the gym, what do you recommend?",
    ]
    pick = st.selectbox("Try one, or type your own below", ["—"] + examples)
    q = st.text_input("Query", value="" if pick == "—" else pick,
                      placeholder="Describe the situation, not the product…")
    compare = st.toggle("Show what today's catalogue would return", value=True)

    if q:
        res = retrieve.search(q, k=3, corpus="enriched", use_filter=True, configs=CONFIGS)
        f = res["filters"]
        st.markdown("**Constraints parsed from that sentence**")
        st.markdown(" ".join(f"<span class='pill'>{d}</span>" for d in (f.describe() or ["none"])),
                    unsafe_allow_html=True)
        cap = (f"SQL filter left {res['filtered_to']} of {len(CARDS)} products; "
               f"embeddings ranked those."
               + (f" ({res['relaxed']})" if res["relaxed"] else ""))
        if res.get("floor") is not None:
            cap += (f" Best match {res['best_score']:.3f} against a decline floor of "
                    f"{res['floor']:.3f}.")
        st.caption(cap)

        if res.get("no_good_match"):
            st.warning(
                "**Nothing in this catalogue is a good match for that.** The best "
                "result scores below the floor calibrated from the simulator — the "
                "similarity at which retrieval has actually found the right product "
                "across 99 test queries. An assistant that answers everything is an "
                "assistant you cannot trust on anything, so this is the honest "
                "answer. Closest available, for reference:")

        if compare:
            left, right = st.columns(2)
            with left:
                st.markdown("##### With Agent Cards")
                for i, h in enumerate(res["hits"], 1):
                    render_hit(h["id"], h["score"], i)
            with right:
                st.markdown("##### Raw catalogue today")
                raw = retrieve.search(q, k=3, corpus="raw", use_filter=False, configs=CONFIGS)
                for i, h in enumerate(raw["hits"], 1):
                    render_hit(h["id"], h["score"], i, show_card=False)
                st.caption("Same embedder, same ranker. The only difference is what the "
                           "product says about itself.")
        else:
            for i, h in enumerate(res["hits"], 1):
                render_hit(h["id"], h["score"], i)



# ------------------------------------------------------------- Compare ----
def _nums(card):
    return {n["key"]: (n.get("value"), n.get("unit", ""))
            for n in card["hard_constraints"].get("numeric", [])}


def _cats(card):
    return {c["key"]: set(map(str, c.get("values", [])))
            for c in card["hard_constraints"].get("categorical", [])}


def _authored(a_card, b_card):
    """Tradeoffs the enricher wrote about this specific pair, in its own words."""
    out = []
    b_title = b_card["identity"]["title"].lower()
    a_title = a_card["identity"]["title"].lower()
    for src, dst, label in ((a_card, b_title, "A"), (b_card, a_title, "B")):
        for c in src.get("comparisons", []):
            against = (c.get("against") or "").lower()
            if against and (against in dst or dst in against
                            or against.split()[-1] in dst):
                out.append((label, c))
    return out


with tab_cmp:
    st.markdown("#### Two products, side by side, on the axis that separates them")
    st.caption("Everything here comes from the Agent Cards already on disk — no "
               "model call. An assistant comparing products for a shopper has to "
               "do exactly this, and it cannot do it from marketing copy, because "
               "marketing copy never says what you give up.")

    cats_available = sorted({CARDS[p]["identity"]["category"] for p in CARDS})
    cat = st.radio("Category", cats_available, horizontal=True,
                   format_func=lambda c: CONFIGS[c].get("label", c))
    pool = sorted([p for p in CARDS if CARDS[p]["identity"]["category"] == cat],
                  key=lambda p: CARDS[p]["identity"]["title"])
    c1, c2 = st.columns(2)
    a_id = c1.selectbox("Product A", pool, index=0,
                        format_func=lambda p: CARDS[p]["identity"]["title"])
    b_id = c2.selectbox("Product B", pool, index=min(1, len(pool) - 1),
                        format_func=lambda p: CARDS[p]["identity"]["title"])

    if a_id == b_id:
        st.info("Pick two different products.")
    else:
        A, B = CARDS[a_id], CARDS[b_id]
        for col, card in ((c1, A), (c2, B)):
            idy = card["identity"]
            col.markdown(
                f"<div class='card'><b>{idy['title']}</b><br>"
                f"<span class='muted'>SGD {idy['price']['amount']:.0f} · "
                f"{idy.get('availability','')}</span><br>"
                f"<span class='muted'>{card['narrative']['one_line_pitch']}</span><br>"
                f"<span class='pill pill-good'>readiness "
                f"{card['readiness']['score']:.0f}</span></div>",
                unsafe_allow_html=True)

        st.markdown("##### Where they actually differ")
        na, nb = _nums(A), _nums(B)
        rows = []
        for key in sorted(set(na) | set(nb)):
            va, unit_a = na.get(key, (None, ""))
            vb, unit_b = nb.get(key, (None, ""))
            unit = unit_a or unit_b
            delta = ""
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = va - vb
                delta = "identical" if d == 0 else f"{d:+g}{unit}"
            rows.append({"attribute": key.replace("_", " "),
                         A["identity"]["title"]: f"{va}{unit}" if va is not None else "—",
                         B["identity"]["title"]: f"{vb}{unit}" if vb is not None else "—",
                         "difference": delta})
        ca, cb = _cats(A), _cats(B)
        same_on_both = []
        for key in sorted(set(ca) | set(cb)):
            sa, sb = ca.get(key, set()), cb.get(key, set())
            if sa == sb:
                same_on_both.append(key.replace("_", " "))
                continue
            rows.append({"attribute": key.replace("_", " "),
                         A["identity"]["title"]: ", ".join(sorted(sa)) or "—",
                         B["identity"]["title"]: ", ".join(sorted(sb)) or "—",
                         "difference": "differs"})
        st.dataframe(rows, hide_index=True)
        if same_on_both:
            st.caption("Identical on both, so not a deciding factor: "
                       + ", ".join(same_on_both) + ".")

        st.markdown("##### What you give up either way")
        authored = _authored(A, B)
        if authored:
            for label, c in authored:
                who = A if label == "A" else B
                st.markdown(
                    f"<div class='card'><b>{who['identity']['title']}</b> — "
                    f"{c.get('direction','')} {c.get('axis','')} than "
                    f"{c.get('against','')}"
                    + (f" ({c['magnitude']})" if c.get("magnitude") else "")
                    + f"<br><span class='muted'>Tradeoff: "
                      f"{c.get('tradeoff','—')}</span></div>",
                    unsafe_allow_html=True)
            st.caption("Written by the enricher against this specific sibling. "
                       "The schema refuses a comparison without a stated tradeoff, "
                       "because \"lighter than X\" on its own does not help anyone rank.")
        else:
            st.caption("Neither card names the other directly — these two were not "
                       "in each other's competitor set at enrichment time. The "
                       "differences above are computed from their typed constraints.")

        ta, tb = set(schema.situational_tags(A)), set(schema.situational_tags(B))
        st.markdown("##### Who each one is for")
        g1, g2, g3 = st.columns(3)
        g1.markdown(f"**Only {A['identity']['title']}**")
        for t in sorted(ta - tb) or ["—"]:
            g1.markdown(f"<span class='pill'>{t}</span>", unsafe_allow_html=True)
        g2.markdown("**Both**")
        for t in sorted(ta & tb) or ["—"]:
            g2.markdown(f"<span class='pill'>{t}</span>", unsafe_allow_html=True)
        g3.markdown(f"**Only {B['identity']['title']}**")
        for t in sorted(tb - ta) or ["—"]:
            g3.markdown(f"<span class='pill'>{t}</span>", unsafe_allow_html=True)

        n1, n2 = st.columns(2)
        for col, card in ((n1, A), (n2, B)):
            col.markdown(f"**Avoid {card['identity']['title']} if**")
            for n in card.get("not_for", [])[:3]:
                col.markdown(f"- {n['exclusion']} — {n['reason']}")

        st.divider()
        st.markdown("##### Which one for a particular shopper?")
        who = st.text_input(
            "Describe the shopper", key="cmp_intent",
            placeholder="e.g. first half marathon, humid weather, tight budget")
        if who:
            ids, vpage, vcard = indexer.load()
            pos = {p: i for i, p in enumerate(ids)}
            qv = embedder.embed([who])[0]
            sa, sb = (float(x) for x in
                      indexer.scores(qv, vpage, vcard, [pos[a_id], pos[b_id]]))
            win, lose = (A, B) if sa >= sb else (B, A)
            ws, ls = (sa, sb) if sa >= sb else (sb, sa)
            f = retrieve.parse_query(who, CONFIGS)
            matched = sorted(set(f.situational) & set(schema.situational_tags(win)))
            st.success(f"**{win['identity']['title']}** — {ws:.3f} against "
                       f"{ls:.3f} for {lose['identity']['title']}.")
            if matched:
                st.caption("Situational tags matched from that sentence: "
                           + ", ".join(matched))
            for u in win.get("use_cases", [])[:2]:
                st.markdown(f"- **{u['scenario']}** — {u['why_it_fits']}")


# ------------------------------------------------------- Before / After ----
with tab_ba:
    pid = st.selectbox("Product", sorted(CARDS),
                       format_func=lambda p: CARDS[p]["identity"]["title"])
    card, base, row = CARDS[pid], BASE[pid], ROWS[pid]
    cfg = CONFIGS[card["identity"]["category"]]
    a, b = st.columns(2)
    with a:
        st.markdown("##### In: the catalogue row")
        st.markdown(f"<div class='card'><b>{row['title']}</b><br>"
                    f"<span class='muted'>{row['description']}</span><br><br>"
                    f"<span class='muted'>tags: {', '.join(row['tags'])}</span><br>"
                    f"<span class='muted'>specs: {row['specs_text'] or '—'}</span></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='big'>{base['readiness']['score']:.0f}</div>"
                    "<span class='muted'>readiness</span>", unsafe_allow_html=True)
        for g in base["readiness"]["top_gaps"]:
            st.markdown(f"<span class='pill pill-bad'>{g}</span>", unsafe_allow_html=True)
    with b:
        st.markdown("##### Out: the Agent Card")
        st.markdown(f"<div class='big'>{card['readiness']['score']:.0f}</div>"
                    "<span class='muted'>readiness</span>", unsafe_allow_html=True)
        comps = card["readiness"]["components"]
        for k, v in comps.items():
            st.progress(min(v, 1.0), text=f"{readiness.LABELS[k]} · {v*100:.0f}%")
        for g in card["readiness"]["top_gaps"]:
            st.markdown(f"<span class='pill pill-bad'>{g}</span>", unsafe_allow_html=True)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Filterable constraints an agent can query**")
        st.json({"numeric": card["hard_constraints"]["numeric"],
                 "categorical": card["hard_constraints"]["categorical"],
                 "situational_tags": card["hard_constraints"]["situational_tags"]},
                expanded=False)
        st.markdown("**Negative information**")
        for n in card["not_for"]:
            st.markdown(f"- **{n['exclusion']}** — {n['reason']} `{n.get('source','')}`")
    with c2:
        st.markdown("**Provenance — every field traces back**")
        st.dataframe([{"field": s["field_path"], "source": s["source"],
                       "evidence": (s.get("evidence") or "")[:70]}
                      for s in card["provenance"]["field_sources"]],
                     hide_index=True)
        if card["provenance"]["unsupported_claims"]:
            st.warning("Flagged for human review before publishing:\n\n"
                       + "\n".join(f"- {u}" for u in card["provenance"]["unsupported_claims"]))
    with st.expander("Full Agent Card JSON"):
        st.json(card)


# ----------------------------------------------------------- Evidence ------
with tab_ev:
    if not REPORT:
        st.info("Run `python -m agentcard simulate` to produce these numbers.")
    else:
        st.markdown("#### Does enrichment change what gets recommended?")
        st.caption("Every query is generated FROM a known product without naming it, "
                   "then retrieval has to find its way back. Same embedder and ranker "
                   "in all three arms — the only variable is what the product says "
                   "about itself, and whether its constraints are typed.")
        r = REPORT["recall_at_k"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw catalogue", f"{r['raw']}%")
        c2.metric("Agent Cards", f"{r['enriched']}%", delta=f"{r['enriched']-r['raw']:+.1f}")
        c3.metric("Agent Cards + SQL filter", f"{r['enriched+sql']}%",
                  delta=f"{r['enriched+sql']-r['enriched']:+.1f} vs semantic only")
        st.caption(f"{REPORT['queries']} generated intent queries · recall@{REPORT['k']} · "
                   f"MRR {REPORT['mrr']['raw']} → {REPORT['mrr']['enriched+sql']}")
        st.markdown("**By category** — the architecture is category-agnostic; "
                    "these are two configs, not two codebases.")
        st.dataframe([{"category": k, **v} for k, v in REPORT["by_category"].items()],
                     hide_index=True)
        if COMPARISON:
            st.divider()
            st.markdown("#### How much of AI-readiness is already in the catalogue?")
            st.caption(COMPARISON["note"])
            vt = COMPARISON["variants"]
            table = []
            for name, label in (("spec_rich", "Spec-rich catalogue"),
                                ("typical", "Typical Shopify export")):
                if name in vt:
                    r = vt[name]["recall_at_k"]
                    table.append({"catalogue": label, "raw": r["raw"],
                                  "enriched": r["enriched"],
                                  "enriched+sql": r["enriched+sql"],
                                  "lift (pts)": vt[name]["lift_points"]})
            st.dataframe(table, hide_index=True)
            if len(table) == 2:
                d = table[1]["lift (pts)"] - table[0]["lift (pts)"]
                st.caption(
                    f"The typical catalogue gains {d:+.1f} points more from enrichment "
                    "than the spec-rich one. A brand already publishing structured specs "
                    "has done part of this work by accident; a brand publishing marketing "
                    "copy and a price has done none of it — and that is most brands.")

        if VALIDATION:
            st.divider()
            st.markdown("#### Does the readiness score predict what gets recommended?")
            st.caption("The brief asks for a score saying how likely an AI is to "
                       "recommend a product. Ours measures content completeness — a "
                       "different claim. So we tested it: per product, readiness "
                       "against how often retrieval actually surfaced it.")
            v = VALIDATION["overall"]
            a, b = st.columns([1, 2])
            a.metric("Spearman rho", f'{v["spearman_rho"]:+.2f}',
                     delta=f'p = {v["p_value"]:.3f}')
            b.info(VALIDATION["interpretation"]["verdict"])
            st.dataframe([{"component": k,
                           "weight": c.get("weight"),
                           "rho vs recall": c.get("spearman_rho"),
                           "p": c.get("p_value"),
                           "note": c.get("note", "")}
                          for k, c in VALIDATION["components"].items()],
                         hide_index=True)
            st.caption(f'{VALIDATION["products"]} products, '
                       f'{VALIDATION["queries_per_product"]} queries each, '
                       f'arm: {VALIDATION["arm"]}. A component that does not predict '
                       "recommendation is measuring something else — often compliance "
                       "value — and that is worth stating rather than hiding.")

        st.divider()
        st.markdown("**Readiness across the catalogue**")
        rows = [{"product": CARDS[p]["identity"]["title"],
                 "category": CARDS[p]["identity"]["category"],
                 "raw": BASE[p]["readiness"]["score"],
                 "enriched": CARDS[p]["readiness"]["score"]} for p in CARDS]
        rows.sort(key=lambda r: r["enriched"])
        chart = {"raw": [x["raw"] for x in rows], "enriched": [x["enriched"] for x in rows]}
        try:
            st.bar_chart(chart, stack=False, height=320)      # side by side, not summed
        except TypeError:
            st.bar_chart(chart, height=320)                   # older streamlit
        st.caption("Bars are side by side, not stacked — the raw baseline is flat at "
                   "23.6 because a catalogue row carries almost nothing an agent can "
                   "reason with, whatever the product is.")
        st.dataframe(rows, hide_index=True)


# --------------------------------------------------------------- Onboard ---
with tab_up:
    st.markdown("#### Bring your own catalogue")
    st.caption("CSV, Excel, Shopify `products.json`, or a PDF lookbook. A PDF page "
               "with no selectable text is rendered and read by a vision model. "
               "Nothing is enriched until you have reviewed what was extracted.")

    src = st.radio("Source", ["Upload a file", "A file already on disk"],
                   horizontal=True, label_visibility="collapsed")
    path = None
    if src == "Upload a file":
        up = st.file_uploader("Catalogue", type=["csv", "tsv", "xlsx", "xls", "json", "pdf"],
                              label_visibility="collapsed")
        if up is not None:
            tmp = config.ROOT / ".cache" / "uploads"
            tmp.mkdir(parents=True, exist_ok=True)
            path = tmp / up.name
            path.write_bytes(up.getbuffer())
    else:
        typed = st.text_input("Path", value=str(config.ROOT / "data" / "raw" /
                                                "typical" / "shopify_facial_skincare.csv"))
        if typed and pathlib.Path(typed).exists():
            path = pathlib.Path(typed)
        elif typed:
            st.error("No file at that path.")

    if path:
        try:
            rows, report = catalog_loader.load(path, "uploaded.catalogue")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file: {type(e).__name__}: {e}")
            st.stop()
        st.session_state["onboard_rows"] = rows

        st.markdown("##### What the loader understood")
        c1, c2, c3 = st.columns(3)
        c1.metric("Products", report["rows"])
        c2.metric("Missing a price", report.get("missing_price", 0))
        c3.metric("Missing a description", report.get("missing_description", 0))
        if report.get("mapped"):
            st.dataframe([{"pipeline field": k, "your column": v}
                          for k, v in report["mapped"].items()], hide_index=True)
        if report.get("unmapped"):
            st.caption("Kept as specification text rather than discarded: "
                       + ", ".join(map(str, report["unmapped"])))
        if report.get("pages_via_vision"):
            st.caption(f'{report["pages_via_vision"]} page(s) had no usable text and '
                       f'were read as images.')

        st.markdown("##### Review before spending anything")
        st.caption("Fix what the loader got wrong. A blank field is a gap to report, "
                   "not an error to hide — the readiness score is supposed to catch it.")
        edited = st.data_editor(
            [{"title": r["title"], "brand": r["brand"], "price": r["price"],
              "description": r["description"][:300], "specs_text": r["specs_text"][:300]}
             for r in rows],
            hide_index=True, num_rows="dynamic", key="onboard_editor")
        for r, e in zip(rows, edited):
            r.update({k: e[k] for k in ("title", "brand", "price") if k in e})

        st.divider()
        st.markdown("##### Find the categories, then write their configs")
        st.caption("Nobody hand-writes a config for a catalogue they have not seen. "
                   "The clusters come from the products; each one's config is derived "
                   "from what that category should carry.")
        if st.button("Cluster and derive configs", type="primary"):
            with st.spinner("Clustering…"):
                clusters = cluster.cluster(rows)
            out = []
            for c in clusters:
                sub = [rows[i] for i in c["product_indices"] if i < len(rows)]
                if not sub:
                    continue
                cat = "auto." + cluster.slug(c["cluster_name"])
                with st.spinner(f'Deriving a config for {c["cluster_name"]}…'):
                    expected = cluster.expected_attributes(c["cluster_name"], sub)
                    cfg = infer_config.infer(sub, c["cluster_name"], cat, expected=expected)
                out.append({"cluster": c["cluster_name"], "products": len(sub),
                            "category": cat, "config": cfg,
                            "issues": infer_config.validate(cfg),
                            "indices": c["product_indices"]})
            st.session_state["onboard_clusters"] = out

        for c in st.session_state.get("onboard_clusters", []):
            with st.expander(f'{c["cluster"]} — {c["products"]} products'):
                cfg = c["config"]
                st.markdown(f'**Required attributes** — numeric: '
                            f'`{", ".join(cfg["required_numeric"]) or "none"}` · '
                            f'categorical: `{", ".join(cfg["required_categorical"]) or "none"}`')
                st.markdown("**Situational vocabulary** — how shopper language reaches "
                            "these specs")
                st.markdown(" ".join(f"<span class='pill'>{t}</span>"
                                     for t in cfg["situational_vocabulary"][:12]) or "—",
                            unsafe_allow_html=True)
                for issue in c["issues"]:
                    st.caption(f"⚠ {issue}")
                st.code(infer_config.to_yaml(cfg), language="yaml")
                fname = f'configs/auto_{cluster.slug(c["cluster"])}.yaml'
                if st.button(f"Save {fname}", key=f'save_{c["category"]}'):
                    p = config.ROOT / fname
                    p.write_text(infer_config.to_yaml(cfg), encoding="utf-8")
                    st.success(f"Wrote {fname} — the pipeline will pick it up on the "
                               f"next run.")

        if st.session_state.get("onboard_clusters"):
            st.divider()
            st.markdown("##### Turn a few into Agent Cards")
            pick = st.selectbox(
                "Cluster", st.session_state["onboard_clusters"],
                format_func=lambda c: f'{c["cluster"]} ({c["products"]})')
            n = st.slider("How many products", 1, min(10, pick["products"]),
                          min(3, pick["products"]))
            st.caption(f"About ${0.018 * n:.2f} at current enrichment prices. "
                       "The full catalogue is a CLI run, not a booth demo.")
            if st.button("Enrich", type="primary"):
                sub = [rows[i] for i in pick["indices"][:n] if i < len(rows)]
                cards = []
                bar = st.progress(0.0)
                for j, row in enumerate(sub, 1):
                    try:
                        cards.append(enrich.enrich_row(row, rows, pick["config"], {}))
                    except Exception as e:  # noqa: BLE001
                        st.error(f'{row["title"]}: {type(e).__name__}: {e}')
                    bar.progress(j / len(sub))
                st.session_state["onboard_cards"] = cards

        for card in st.session_state.get("onboard_cards", []):
            r = card.get("readiness", {})
            with st.expander(f'{card["identity"]["title"]} — readiness '
                             f'{r.get("score", 0):.0f}'):
                t1, t2, t3 = st.tabs(["Paste into your product page",
                                      "schema.org JSON-LD", "Needs review"])
                t1.text_area("copy", copy_out.passage(card), height=260,
                             key=f'cp_{card["id"]}', label_visibility="collapsed")
                t2.code(json.dumps(copy_out.json_ld(card), indent=2), language="json")
                notes = copy_out.review_notes(card)
                if notes:
                    for note in notes[:12]:
                        t3.markdown(f"- {note}")
                else:
                    t3.caption("Nothing flagged — every field traces to the catalogue.")
                for g in r.get("top_gaps", []):
                    st.markdown(f"<span class='pill pill-bad'>{g}</span>",
                                unsafe_allow_html=True)
