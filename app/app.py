import hashlib
import json

import streamlit as st
import pandas as pd

from extraction import load_catalog_file
from normalization import normalize_cluster
from fit import generate_fit_criteria, evaluate_fit
import gaps
import export
from ui import RunLog
from retrieval import (
    build_index,
    product_to_raw_text,
    product_to_optimized_text,
    run_query,
)
from pipeline import (
    cluster_products,
    determine_expected_attributes,
    attribute_completeness,
    suggest_personas,
    generate_user_story,
    generate_product_content,
    readiness_score,
)

st.set_page_config(page_title="Brand Amplifier", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "products": None,          # list[dict] confirmed products
    "clusters": None,          # list[{cluster_name, product_indices}]
    "cluster_data": {},        # cluster_name -> {expected_attrs, avg_completeness, per_product, missing_counts, personas, selected_persona, user_story, content: {product_name: text}}
    "analysis_done": False,
    "query_input": "I'm training for a half marathon in Singapore's humid weather and need lightweight shoes under S$200.",
    "index_signature": None,   # invalidates the cached indexes when the catalog/content changes
    "raw_index": None,
    "optimized_index": None,
    "query_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🧭 Agent Readiness Copilot")
st.caption("Upload a product catalog → find AI-recommendation gaps → generate persona-driven, agent-optimized content.")

tab1, tab_ask, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Upload & Extract",
    "🔎 Ask like a shopper",
    "2️⃣ Dashboard",
    "3️⃣ Personas",
    "4️⃣ Generated Content",
])

# ---------------------------------------------------------------------------
# TAB 1 — Upload & Extract (+ confirm/edit merged in here)
# ---------------------------------------------------------------------------
with tab1:
    st.info("Uploaded content is sent to OpenAI's API for processing.", icon="ℹ️")

    uploaded_file = st.file_uploader("Upload catalog (PDF, CSV, XLSX, or JSON)", type=["pdf", "csv", "xlsx", "xls", "json"])

    if uploaded_file is not None and st.button("Extract products", type="primary"):
        log = RunLog(f"Extracting from {uploaded_file.name}")
        log.step(0.02, f"Reading {uploaded_file.name}")

        def progress_cb(done, total, mode):
            reason = "image-only page, reading it visually" if mode == "vision" else "selectable text"
            log.step(done / total, f"Page {done}/{total} — {reason}")

        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                products = load_catalog_file(uploaded_file, progress_callback=progress_cb)
            else:
                products = load_catalog_file(uploaded_file)
            st.session_state["products"] = products
            st.session_state["clusters"] = None
            st.session_state["cluster_data"] = {}
            st.session_state["analysis_done"] = False
            modes = {}
            for p in products:
                m = p.get("extraction_mode", "table")
                modes[m] = modes.get(m, 0) + 1
            log.note(f"Found {len(products)} products ({', '.join(f'{v} via {k}' for k, v in modes.items())})")
            log.done(f"Extracted {len(products)} products")
        except Exception as e:
            log.fail(str(e))

    if st.session_state["products"]:
        st.subheader("Extracted products — review & edit before analysis")
        st.caption("Fix anything the extractor got wrong. Blank fields are treated as gaps, not errors.")

        df = pd.DataFrame([
            {
                "name": p.get("name", ""),
                "price": p.get("price", ""),
                "description": p.get("description", ""),
                # JSON, not "k: v, k: v" - a comma inside any value (an ingredient list,
                # "258g (US M9), wide") used to be split into garbage on the way back.
                "specs": json.dumps(p.get("specs", {}), ensure_ascii=False),
            }
            for p in st.session_state["products"]
        ])
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="edit_products")

        if st.button("Confirm & run clustering + gap analysis", type="primary"):
            # rebuild products list from edited dataframe
            def clean_str(val):
                # pandas turns blank/edited-out cells into NaN (a float), not "".
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return ""
                return str(val).strip()

            rebuilt = []
            for _, row in edited_df.iterrows():
                specs = {}
                specs_raw = clean_str(row["specs"])
                if specs_raw:
                    try:
                        parsed = json.loads(specs_raw)
                        if isinstance(parsed, dict):
                            specs = {str(k): str(v) for k, v in parsed.items()}
                    except (json.JSONDecodeError, TypeError):
                        # someone hand-typed "k: v, k: v" over the JSON - accept it
                        for pair in specs_raw.split(","):
                            if ":" in pair:
                                k, v = pair.split(":", 1)
                                specs[k.strip()] = v.strip()
                rebuilt.append({
                    "name": clean_str(row["name"]) or "Unnamed product",
                    "price": clean_str(row["price"]) or None,
                    "description": clean_str(row["description"]),
                    "specs": specs,
                })
            st.session_state["products"] = rebuilt

            log = RunLog(f"Analyzing {len(rebuilt)} products")
            log.step(0.03, f"Clustering {len(rebuilt)} products by similarity")
            clusters = cluster_products(rebuilt)
            st.session_state["clusters"] = clusters
            log.note(f"Found {len(clusters)} clusters: " +
                     ", ".join(f"{c['cluster_name']} ({len(c['product_indices'])})" for c in clusters))

            cluster_data = {}
            SPAN, BASE = 0.92, 0.05
            for c_idx, c in enumerate(clusters):
                    name = c["cluster_name"]
                    members = [rebuilt[i] for i in c["product_indices"] if i < len(rebuilt)]
                    slot = SPAN / max(len(clusters), 1)
                    start = BASE + c_idx * slot
                    log.step(start, f"[{name}] inferring attribute schema")

                    def _norm_progress(done, total_batches, cluster_name, _s=start, _sl=slot):
                        log.step(_s + _sl * 0.15 + (_sl * 0.45) * (done / max(total_batches, 1)),
                                 f"[{cluster_name}] normalizing batch {done + 1}/{total_batches}")

                    norm_stats = normalize_cluster(name, members, progress=_norm_progress)
                    if norm_stats["schema"]:
                        log.note(f"[{name}] schema: {', '.join(norm_stats['schema'])}")
                        log.note(f"[{name}] normalized {norm_stats['normalized_count']}/{norm_stats['total']}"
                                 + (f", dropped {norm_stats['rejected_values']} unverifiable value(s)"
                                    if norm_stats["rejected_values"] else "")
                                 + (f", {norm_stats['failed_batches']} batch(es) failed - raw specs kept"
                                    if norm_stats["failed_batches"] else ""))
                    else:
                        log.note(f"[{name}] normalization skipped - falling back to raw specs")

                    # The inferred schema IS the expected-attribute list. If normalization
                    # was skipped, fall back to asking for expected attributes the old way.
                    expected_attrs = norm_stats["schema"] or determine_expected_attributes(name, members)
                    avg_completeness, per_product, missing_counts = attribute_completeness(members, expected_attrs)
                    log.note(f"[{name}] attribute completeness {avg_completeness}%")

                    log.step(start + slot * 0.65, f"[{name}] generating persona candidates")
                    personas = suggest_personas(name, expected_attrs, per_product)

                    # Turn each persona's stated need into machine-checkable criteria, then
                    # evaluate every product in plain Python. Splits the old single rating
                    # into coverage (a content problem) and fit (a merchandising one).
                    for p_idx, persona in enumerate(personas):
                        log.step(start + slot * (0.75 + 0.2 * (p_idx / max(len(personas), 1))),
                                 f"[{name}] fit criteria for '{persona.get('title','?')}'")
                        criteria = generate_fit_criteria(persona, expected_attrs, members)
                        persona["fit"] = evaluate_fit(criteria, members) if criteria else None
                        if persona["fit"]:
                            pf = persona["fit"]
                            log.note(f"[{name}] '{persona.get('title','?')}' - coverage {pf['coverage_pct']}%, "
                                     f"fit {pf['fit_pct']}% ({pf['qualifying']}/{pf['total']} qualify)")
                        else:
                            log.note(f"[{name}] '{persona.get('title','?')}' - no criteria generated")

                    cluster_data[name] = {
                        "product_indices": c["product_indices"],
                        "members": members,
                        "expected_attrs": expected_attrs,
                        "norm_stats": norm_stats,
                        "avg_completeness": avg_completeness,
                        "per_product": per_product,
                        "missing_counts": missing_counts,
                        "personas": personas,
                        "selected_persona": None,
                        "user_story": None,
                        "content": {},
                    }
            st.session_state["cluster_data"] = cluster_data
            st.session_state["analysis_done"] = True
            log.done(f"Analyzed {len(rebuilt)} products across {len(clusters)} clusters")
            st.success("Analysis complete — see the Dashboard and Personas tabs.")

# ---------------------------------------------------------------------------
# TAB "Ask like a shopper" — the evidence step
# ---------------------------------------------------------------------------
# The rest of the app PRODUCES agent-optimized content. This tab TESTS it: a real
# intent-driven query is run against the catalog twice — over the original content and
# over the generated content — so the difference in what an assistant would recommend
# is visible side by side, rather than asserted.

EXAMPLE_QUERIES = [
    "I'm training for a half marathon in Singapore's humid weather and need lightweight shoes under S$200.",
    "Find me a sustainable skincare routine for oily skin that takes less than 5 minutes every morning.",
    "I need something hard-wearing for daily commuting that still looks smart enough for the office.",
]


def _set_query(text: str):
    st.session_state["query_input"] = text


def _content_signature(cluster_data: dict) -> str:
    """Changes whenever the catalog or the generated content changes, so the cached
    embeddings are rebuilt instead of silently going stale."""
    parts = []
    for cluster_name, data in cluster_data.items():
        for product in data.get("members", []):
            parts.append(f"{cluster_name}|{product.get('name')}|{len(str(product.get('specs') or {}))}")
        for product_name, text in (data.get("content") or {}).items():
            parts.append(f"{cluster_name}|{product_name}|gen{len(text)}")
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _build_indexes(cluster_data: dict):
    """Both indexes cover EXACTLY the same products — those that have generated content —
    so the only thing that differs between Before and After is the content itself.

    Including un-optimized products in the raw index only would rig the comparison: the
    Before side would search the whole catalog while the After side searched a subset,
    and the After side would be forced to return whatever it had regardless of fit."""
    raw_entries, optimized_entries = [], []
    for cluster_name, data in cluster_data.items():
        content_map = data.get("content") or {}
        for product in data.get("members", []):
            name = str(product.get("name") or "Unnamed product")
            generated = content_map.get(name)
            if not generated:
                continue  # excluded from BOTH sides, never just one
            key = f"{cluster_name}::{name}"
            raw_entries.append({
                "key": key, "name": name, "cluster": cluster_name,
                "text": product_to_raw_text(product),
            })
            optimized_entries.append({
                "key": key, "name": name, "cluster": cluster_name,
                "text": product_to_optimized_text(product, generated),
            })
    return build_index(raw_entries), build_index(optimized_entries)


def _coverage(cluster_data: dict):
    """(products with generated content, products in catalog)"""
    total = sum(len(data.get("members", [])) for data in cluster_data.values())
    covered = sum(len(data.get("content") or {}) for data in cluster_data.values())
    return covered, total


def _generate_all(cluster_data: dict, progress=None):
    """Fill in content for every cluster that doesn't have it yet, so a query can be run
    against the whole catalog. Uses each cluster's selected persona, or its highest-rated
    candidate if none was chosen by hand."""
    pending = [(n, d) for n, d in cluster_data.items()
               if len(d.get("content") or {}) < len(d.get("members", []))]
    for done, (cluster_name, data) in enumerate(pending):
        if progress:
            progress(done, len(pending), cluster_name)
        personas = data.get("personas") or []
        persona = data.get("selected_persona")
        if not persona:
            if not personas:
                continue
            persona = max(personas, key=lambda p: p.get("persona_rating_pct", 0))
            data["selected_persona"] = persona
        if not data.get("user_story"):
            data["user_story"] = generate_user_story(cluster_name, persona)
        content = dict(data.get("content") or {})
        for product in data.get("members", []):
            if product.get("name") in content:
                continue
            content[product["name"]] = generate_product_content(
                product, cluster_name, persona, data["user_story"]
            )
        data["content"] = content


def _render_hits(hits: list, empty_message: str):
    if not hits:
        st.caption(empty_message)
        return
    for hit in hits:
        marker = "✅" if hit.get("recommend") else "⚠️"
        st.markdown(f"**{hit['final_rank']}. {hit['name']}** {marker}")
        st.caption(f"{hit.get('cluster','')} · similarity {hit['score']}")
        if hit.get("reason"):
            st.write(hit["reason"])
        if hit.get("cited_attributes"):
            st.markdown("**Cited:** " + ", ".join(f"`{a}`" for a in hit["cited_attributes"]))
        if hit.get("unanswered"):
            st.markdown("**Content can't answer:** " + ", ".join(hit["unanswered"]))
        st.divider()


with tab_ask:
    st.subheader("Ask like a shopper")
    st.caption(
        "Type a real intent-driven question — the kind someone would ask an AI assistant. "
        "It is run against your catalog twice: once over the content as you supplied it, "
        "once over the generated content. The two result sets sit side by side."
    )

    if not st.session_state["analysis_done"]:
        st.info("Run extraction + analysis in Tab 1 first.")
    else:
        cd = st.session_state["cluster_data"]
        has_content = any(data.get("content") for data in cd.values())

        covered, total = _coverage(cd)

        if not has_content:
            st.warning(
                "No generated content yet — pick a persona and click Generate in the Personas tab, "
                "or generate for every cluster below. Until then there is nothing to compare "
                "your original catalog against.",
                icon="⚠️",
            )

        if covered < total:
            st.warning(
                f"**{covered} of {total} products have generated content.** Both sides of the "
                f"comparison are limited to those {covered}, so the only difference between them "
                f"is the content itself. To query your whole catalog, generate the rest — "
                f"otherwise a query about a product you haven't optimized has nothing to match.",
                icon="⚖️",
            )
            if st.button(f"Generate content for the remaining {total - covered} products"):
                log = RunLog(f"Generating content for {total - covered} products")

                def _progress(done, total_clusters, cluster_name):
                    log.step(done / max(total_clusters, 1),
                             f"Cluster {done + 1}/{total_clusters} — {cluster_name}")

                try:
                    _generate_all(cd, progress=_progress)
                    now_covered, now_total = _coverage(cd)
                    log.note(f"Coverage now {now_covered}/{now_total} products")
                    log.done(f"Generated content — {now_covered}/{now_total} products covered")
                    st.session_state["index_signature"] = None   # force a rebuild
                    st.rerun()
                except Exception as e:
                    log.fail(str(e))
        elif has_content:
            st.caption(f"Comparing all {total} products in the catalog.")

        st.text_input("What is the shopper asking for?", key="query_input")

        st.caption("Or try one of these:")
        example_cols = st.columns(len(EXAMPLE_QUERIES))
        for i, (col, example) in enumerate(zip(example_cols, EXAMPLE_QUERIES)):
            with col:
                st.button(
                    example[:46] + "…",
                    key=f"example_query_{i}",
                    on_click=_set_query,
                    args=(example,),
                )

        if st.button("Run query", type="primary", disabled=not has_content):
            query = st.session_state["query_input"].strip()
            if not query:
                st.error("Enter a question first.")
            else:
                log = RunLog("Running query")
                try:
                    signature = _content_signature(cd)
                    if signature != st.session_state["index_signature"]:
                        log.step(0.1, "Embedding the catalog twice (original + generated content)")
                        raw_index, optimized_index = _build_indexes(cd)
                        st.session_state["raw_index"] = raw_index
                        st.session_state["optimized_index"] = optimized_index
                        st.session_state["index_signature"] = signature
                        log.note(f"Indexed {len(raw_index['entries'])} products on each side")
                    else:
                        log.step(0.1, "Reusing cached embeddings — catalog unchanged")

                    log.step(0.35, "Embedding the query")
                    log.step(0.5, "Searching both indexes and ranking with citations")
                    result = run_query(
                        st.session_state["raw_index"],
                        st.session_state["optimized_index"],
                        query,
                    )
                    st.session_state["query_result"] = result
                    moved = [m for m in result["movement"] if m["before"] != m["after"]]
                    log.note(f"{len(result['raw'])} hits before, {len(result['optimized'])} after; "
                             f"{len(moved)} product(s) changed rank")
                    log.done("Query complete")
                except Exception as e:
                    st.session_state["query_result"] = None
                    log.fail(str(e))

        result = st.session_state.get("query_result")
        if result:
            st.divider()
            st.markdown(f"**Query:** _{result['query']}_")

            moves = [m for m in result["movement"] if m["before"] != m["after"]]
            if moves:
                lines = []
                for m in moves[:3]:
                    before = "not surfaced at all" if m["before"] is None else f"#{m['before']}"
                    lines.append(f"**{m['name']}**: {before} → #{m['after']}")
                st.success("  ·  ".join(lines), icon="📈")

            col_before, col_after = st.columns(2)
            with col_before:
                st.markdown("### Before")
                st.caption("Ranked using the catalog content as supplied")
                _render_hits(result["raw"], "Nothing retrieved from the original content.")
            with col_after:
                st.markdown("### After")
                st.caption("Ranked using the generated agent-optimized content")
                _render_hits(result["optimized"], "Nothing retrieved — generate content first.")

# ---------------------------------------------------------------------------
# TAB 2 — Dashboard
# ---------------------------------------------------------------------------
with tab2:
    if not st.session_state["analysis_done"]:
        st.info("Run extraction + analysis in Tab 1 first.")
    else:
        cd = st.session_state["cluster_data"]

        # Overall headline score = average of cluster readiness scores
        cluster_scores = []
        for name, data in cd.items():
            persona_rating = data["selected_persona"]["persona_rating_pct"] if data["selected_persona"] else None
            score = readiness_score(data["avg_completeness"], persona_rating)
            cluster_scores.append(score)
        overall = round(sum(cluster_scores) / len(cluster_scores), 1) if cluster_scores else 0.0

        st.metric("Overall Catalog Readiness Score", f"{overall}%",
                   help="30% attribute completeness + 70% rating of the selected persona, averaged across clusters.")

        st.divider()
        head = gaps.headline(cd)
        st.subheader("What your catalog can't answer")
        st.caption(
            "Generated content is grounded in verified attributes, so it can never invent a value "
            "nobody supplied. For genuinely missing data the fix isn't better copy — it's this list, "
            "handed to whoever owns the product data."
        )
        if head["attributes_needed"] == 0:
            st.success("Nothing missing — every product carries its full category schema.")
        else:
            worst = head["worst"]
            st.markdown(
                f"**{head['attributes_needed']} attributes** need supplying across "
                f"**{head['products_affected']} of {head['total_products']} products**. "
                f"Biggest single gap: `{worst['attribute']}`, missing on "
                f"{worst['missing_count']}/{worst['total_products']} in {worst['cluster']}."
            )
            attr_rows = gaps.attribute_requests(cd)
            prod_rows = gaps.product_requests(cd)
            st.dataframe(
                pd.DataFrame(attr_rows)[["cluster", "attribute", "missing_count",
                                          "missing_pct", "needed_because"]],
                use_container_width=True, hide_index=True,
            )
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    "Download data request (by attribute)",
                    pd.DataFrame(attr_rows).to_csv(index=False),
                    file_name="data_request_by_attribute.csv", mime="text/csv",
                )
            with d2:
                st.download_button(
                    "Download data request (by product)",
                    pd.DataFrame(prod_rows).to_csv(index=False),
                    file_name="data_request_by_product.csv", mime="text/csv",
                )

        st.divider()
        st.subheader("Per-cluster breakdown")

        for name, data in cd.items():
            persona = data["selected_persona"]
            pf = (persona or {}).get("fit")
            persona_coverage = pf["coverage_pct"] if pf else None
            score = readiness_score(data["avg_completeness"], persona_coverage)

            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                with c1:
                    st.markdown(f"### {name}")
                    st.caption(f"{len(data['members'])} products")
                with c2:
                    st.metric("Attribute completeness", f"{data['avg_completeness']}%")
                with c3:
                    st.metric("Readiness score", f"{score}%",
                              help="Content-fixable coverage only. Fit is reported separately.")
                with c4:
                    if pf:
                        st.metric("Persona fit", f"{pf['fit_pct']}%",
                                  help=f"{pf['qualifying']} of {pf['total']} products satisfy every "
                                       "criterion. A merchandising fact, not a content gap — "
                                       "deliberately excluded from the readiness score.")
                    else:
                        st.metric("Persona fit", "—")

                ns = data.get("norm_stats") or {}
                if ns.get("schema"):
                    bits = [f"**{ns['normalized_count']}/{ns['total']}** products normalized"]
                    if ns.get("rejected_values"):
                        bits.append(f"{ns['rejected_values']} unverifiable value(s) dropped")
                    if ns.get("failed_batches"):
                        bits.append(f"{ns['failed_batches']} batch(es) failed - raw specs kept")
                    st.caption("Knowledge layer: " + " · ".join(bits))
                else:
                    st.caption("Knowledge layer: not applied - using raw specs and inferred attributes.")

                missing = [attr for attr, count in data["missing_counts"].items() if count > 0]
                if missing:
                    st.markdown(f"**Missing attributes:** {', '.join(missing)}")
                else:
                    st.markdown("**Missing attributes:** none 🎉")

                if persona:
                    st.markdown(f"**Selected persona:** {persona['title']}")
                    if pf:
                        st.caption(
                            f"Coverage {pf['coverage_pct']}% — how much of this shopper's criteria "
                            f"the data can answer.  ·  Fit {pf['fit_pct']}% — "
                            f"{pf['qualifying']}/{pf['total']} products actually qualify."
                        )
                        with st.expander("Criteria and per-product verdicts"):
                            for c in pf["criteria"]:
                                st.markdown(
                                    f"- `{c['attribute']} {c['operator']} {c['value']}`"
                                    + (f" — {c['rationale']}" if c["rationale"] else "")
                                )
                            st.divider()
                            for r in pf["per_product"]:
                                mark = "✅" if r["qualifies"] else "—"
                                st.write(f"{mark} **{r['name']}**")
                                if r["failed"]:
                                    st.caption("   fails: " + ", ".join(r["failed"]))
                                if r["unknown"]:
                                    st.caption("   no data: " + ", ".join(r["unknown"]))
                else:
                    st.markdown("**Selected persona:** none yet — pick one in the Personas tab.")

                with st.expander("Per-product attribute detail"):
                    by_name = {str(m.get("name")): m for m in data["members"]}
                    for p in data["per_product"]:
                        st.write(
                            f"- **{p['name']}** — {p['completeness_pct']}% "
                            f"(missing: {', '.join(p['missing']) if p['missing'] else 'none'})"
                        )
                        norm = (by_name.get(p["name"], {}) or {}).get("specs_normalized")
                        if norm:
                            st.caption("   extracted: " + " · ".join(f"`{k}`: {v}" for k, v in norm.items()))

# ---------------------------------------------------------------------------
# TAB 3 — Personas
# ---------------------------------------------------------------------------
with tab3:
    if not st.session_state["analysis_done"]:
        st.info("Run extraction + analysis in Tab 1 first.")
    else:
        cd = st.session_state["cluster_data"]

        for name, data in cd.items():
            st.markdown(f"## {name}")
            personas = data["personas"]

            if not personas:
                st.warning("No persona candidates generated for this cluster.")
                continue

            def _label(p):
                pf = p.get("fit")
                if pf:
                    return (f"{p['title']} — coverage {pf['coverage_pct']}%, "
                            f"fit {pf['fit_pct']}% ({pf['qualifying']}/{pf['total']} qualify)")
                return f"{p['title']} — coverage {p['persona_rating_pct']}% (no criteria generated)"

            options = [_label(p) for p in personas]

            current_selection = data["selected_persona"]["title"] if data["selected_persona"] else None
            default_index = 0
            if current_selection:
                for i, p in enumerate(personas):
                    if p["title"] == current_selection:
                        default_index = i
                        break

            choice = st.radio(
                f"Choose a persona for '{name}'",
                options=range(len(personas)),
                format_func=lambda i: options[i],
                index=default_index,
                key=f"persona_choice_{name}",
            )

            chosen_persona = personas[choice]
            with st.expander("Persona detail"):
                st.write(f"**Narrative seed:** {chosen_persona.get('narrative_seed','')}")
                st.write(f"**Supporting attributes needed:** {', '.join(chosen_persona.get('supporting_attributes', []))}")
                st.write(f"**Covered by current data:** {', '.join(chosen_persona.get('covered_attributes', [])) or 'none'}")
                st.write(f"**Still missing:** {', '.join(chosen_persona.get('missing_attributes', [])) or 'none'}")

            if st.button(f"Generate story & content for '{name}'", key=f"generate_{name}", type="primary"):
                log = RunLog(f"Writing content for {name}")
                try:
                    log.step(0.05, f"Writing the user story for '{chosen_persona['title']}'")
                    story = generate_user_story(name, chosen_persona)
                    data["selected_persona"] = chosen_persona
                    data["user_story"] = story
                    log.note(f"Story: {story}")

                    content_map = {}
                    total = len(data["members"])
                    for i, product in enumerate(data["members"]):
                        pname = str(product.get("name", "Unnamed product"))
                        grounded = "verified attributes" if product.get("specs_normalized") else "raw specs"
                        log.step(0.1 + 0.9 * (i / max(total, 1)),
                                 f"{i + 1}/{total} — {pname} (from {grounded})")
                        content_map[pname] = generate_product_content(product, name, chosen_persona, story)
                    data["content"] = content_map
                    log.done(f"Wrote content for {total} products in {name}")
                    st.success("Story + content generated — see the Generated Content tab.")
                except Exception as e:
                    log.fail(str(e))

            if data["user_story"]:
                st.markdown(f"**Current user story:** _{data['user_story']}_")

            st.divider()

# ---------------------------------------------------------------------------
# TAB 4 — Generated Content
# ---------------------------------------------------------------------------
with tab4:
    if not st.session_state["analysis_done"]:
        st.info("Run extraction + analysis in Tab 1 first.")
    else:
        cd = st.session_state["cluster_data"]
        any_content = any(data["content"] for data in cd.values())

        if not any_content:
            st.info("No content generated yet — pick a persona and click Generate in the Personas tab.")
        else:
            st.subheader("Export")
            st.caption(
                "Three formats. JSON-LD is the one to point a brand at — schema.org `Product` markup "
                "drops straight into a product page, so the verified attributes become machine-readable "
                "at the source instead of living in this app."
            )
            detected = export.detect_currency(cd)
            currency = st.text_input(
                "Currency code for JSON-LD offers",
                value=detected or "SGD",
                help="Detected from your price strings where possible; schema.org Offer needs an "
                     "ISO code, which most catalog exports don't carry.",
            ).strip().upper() or None

            e1, e2, e3 = st.columns(3)
            with e1:
                st.download_button("CSV", export.to_csv(cd),
                                   file_name="agent_optimized_catalog.csv", mime="text/csv")
            with e2:
                st.download_button("JSON", export.to_json(cd),
                                   file_name="agent_optimized_catalog.json", mime="application/json")
            with e3:
                st.download_button("schema.org JSON-LD", export.to_jsonld(cd, currency),
                                   file_name="agent_optimized_catalog.jsonld",
                                   mime="application/ld+json")

            with st.expander("Preview the JSON-LD for one product"):
                st.code(json.dumps(
                    json.loads(export.to_jsonld(cd, currency))["@graph"][0], indent=2
                ), language="json")

            st.divider()
            for name, data in cd.items():
                if not data["content"]:
                    continue
                st.markdown(f"## {name}")
                if data["user_story"]:
                    st.caption(f"Persona story: {data['user_story']}")

                for product_name, text in data["content"].items():
                    with st.expander(product_name, expanded=False):
                        st.text_area(
                            "Copy-pastable content",
                            value=text,
                            height=220,
                            key=f"content_{name}_{product_name}",
                        )
                st.divider()
