import streamlit as st
import pandas as pd

from extraction import load_catalog_file
from pipeline import (
    cluster_products,
    determine_expected_attributes,
    attribute_completeness,
    suggest_personas,
    generate_user_story,
    competitors_for,
    generate_agent_content,
    build_raw_passage,
    render_passage,
    readiness_score,
    score_components,
    top_gaps,
    ask_confidence,
    LABELS,
)
from llm_utils import get_spend_summary

st.set_page_config(page_title="Brand Enabler", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
defaults = {
    "products": None,
    "clusters": None,
    "cluster_data": {},   # cluster_name -> see shape built in Tab 1
    "analysis_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🧭 Brand Enabler")
st.caption("Upload a product catalog → find AI-recommendation gaps → generate persona-driven, agent-optimized content.")

with st.sidebar:
    spend = get_spend_summary()
    st.caption(f"API spend this environment: ${spend.get('usd', 0):.2f} "
               f"({spend.get('calls', 0)} calls, {spend.get('cached', 0)} cached)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1️⃣ Upload & Extract", "2️⃣ Dashboard", "3️⃣ Personas", "4️⃣ Generated Content", "5️⃣ Ask"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Upload & Extract (+ confirm/edit merged in here)
# ---------------------------------------------------------------------------
with tab1:
    st.info("Uploaded content is sent to OpenAI's API for processing.", icon="ℹ️")

    uploaded_file = st.file_uploader("Upload catalog (PDF, CSV, XLSX, or JSON)",
                                      type=["pdf", "csv", "xlsx", "xls", "json"])

    if uploaded_file is not None and st.button("Extract products", type="primary"):
        progress_bar = st.progress(0.0, text="Starting extraction...")

        def progress_cb(done, total, mode):
            progress_bar.progress(done / total, text=f"Page {done}/{total} (using {mode} extraction)...")

        with st.spinner("Extracting products..."):
            try:
                if uploaded_file.name.lower().endswith(".pdf"):
                    products = load_catalog_file(uploaded_file, progress_callback=progress_cb)
                else:
                    products = load_catalog_file(uploaded_file)
                st.session_state["products"] = products
                st.session_state["clusters"] = None
                st.session_state["cluster_data"] = {}
                st.session_state["analysis_done"] = False
                progress_bar.progress(1.0, text="Done.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

    if st.session_state["products"]:
        st.subheader("Extracted products — review & edit before analysis")
        st.caption("Fix anything the extractor got wrong. Blank fields are treated as gaps, not errors.")

        df = pd.DataFrame([
            {
                "name": p.get("name", ""),
                "price": p.get("price", ""),
                "description": p.get("description", ""),
                "specs": ", ".join(f"{k}: {v}" for k, v in (p.get("specs") or {}).items()),
            }
            for p in st.session_state["products"]
        ])
        edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="edit_products")

        if st.button("Confirm & run clustering + gap analysis", type="primary"):
            def clean_str(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return ""
                return str(val).strip()

            rebuilt = []
            for _, row in edited_df.iterrows():
                specs = {}
                specs_raw = clean_str(row["specs"])
                if specs_raw:
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

            with st.spinner("Clustering products by similarity..."):
                clusters = cluster_products(rebuilt)
                st.session_state["clusters"] = clusters

            cluster_data = {}
            with st.spinner("Analyzing attribute gaps and generating persona candidates..."):
                for c in clusters:
                    name = c["cluster_name"]
                    members = [rebuilt[i] for i in c["product_indices"] if i < len(rebuilt)]
                    expected_attrs = determine_expected_attributes(name, members)
                    avg_completeness, per_product, missing_counts = attribute_completeness(members, expected_attrs)
                    personas = suggest_personas(name, expected_attrs, per_product)
                    cluster_data[name] = {
                        "product_indices": c["product_indices"],
                        "members": members,
                        "expected_attrs": expected_attrs,
                        "avg_completeness": avg_completeness,
                        "per_product": per_product,
                        "missing_counts": missing_counts,
                        "personas": personas,
                        "selected_persona": None,
                        "user_story": None,
                        "agent_content": {},   # product_name -> structured dict from generate_agent_content
                        "content": {},         # product_name -> rendered copy-pastable text
                    }
            st.session_state["cluster_data"] = cluster_data
            st.session_state["analysis_done"] = True
            st.success("Analysis complete — see the Dashboard and Personas tabs.")

# ---------------------------------------------------------------------------
# TAB 2 — Dashboard (incl. Before / After)
# ---------------------------------------------------------------------------
with tab2:
    if not st.session_state["analysis_done"]:
        st.info("Run extraction + analysis in Tab 1 first.")
    else:
        cd = st.session_state["cluster_data"]

        def _product_completeness_pct(data, product_name):
            for p in data["per_product"]:
                if p["name"] == product_name:
                    return p["completeness_pct"]
            return data["avg_completeness"]

        def _cluster_score(data):
            scores = []
            for p in data["per_product"]:
                agent_content = data["agent_content"].get(p["name"])
                scores.append(readiness_score(p["completeness_pct"], agent_content))
            return round(sum(scores) / len(scores), 1) if scores else 0.0

        cluster_scores = [_cluster_score(data) for data in cd.values()]
        overall = round(sum(cluster_scores) / len(cluster_scores), 1) if cluster_scores else 0.0

        st.metric("Overall Catalog Readiness Score", f"{overall}%",
                   help="5 weighted components computed deterministically, averaged per product then per cluster.")

        st.divider()
        st.subheader("Per-cluster breakdown")

        for name, data in cd.items():
            score = _cluster_score(data)
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.markdown(f"### {name}")
                    st.caption(f"{len(data['members'])} products")
                with c2:
                    st.metric("Attribute completeness", f"{data['avg_completeness']}%")
                with c3:
                    st.metric("Readiness score", f"{score}%")

                missing = [attr for attr, count in data["missing_counts"].items() if count > 0]
                st.markdown(f"**Missing attributes:** {', '.join(missing) if missing else 'none 🎉'}")

                if data["selected_persona"]:
                    st.markdown(f"**Selected persona:** {data['selected_persona']['title']}")
                else:
                    st.markdown("**Selected persona:** none yet — pick one in the Personas tab.")

                with st.expander("Per-product detail"):
                    for p in data["per_product"]:
                        agent_content = data["agent_content"].get(p["name"])
                        p_score = readiness_score(p["completeness_pct"], agent_content)
                        st.write(f"- **{p['name']}** — readiness {p_score}% "
                                 f"(attribute completeness {p['completeness_pct']}%, "
                                 f"missing: {', '.join(p['missing']) if p['missing'] else 'none'})")

        st.divider()
        st.subheader("Before / After — one product, raw vs. generated")

        all_products = []
        for cname, data in cd.items():
            for p in data["members"]:
                all_products.append((cname, p))

        if not all_products:
            st.info("No products available.")
        else:
            options = [f"{p['name']}  ·  {cname}" for cname, p in all_products]
            idx = st.selectbox("Product", range(len(options)), format_func=lambda i: options[i])
            cname, product = all_products[idx]
            data = cd[cname]
            completeness_pct = _product_completeness_pct(data, product["name"])
            agent_content = data["agent_content"].get(product["name"])

            a, b = st.columns(2)
            with a:
                st.markdown("##### In: the catalog row")
                with st.container(border=True):
                    st.markdown(f"**{product['name']}**")
                    st.caption(product.get("description") or "—")
                    st.caption(f"Price: {product.get('price') or '—'}")
                    st.caption(f"Specs: {product.get('specs') or '—'}")
                before_score = readiness_score(completeness_pct, None)
                st.markdown(f"### {before_score}%")
                st.caption("readiness — before")
                for g in top_gaps(completeness_pct, None, [a for a in data["expected_attrs"]
                                                            if a not in next(
                                                                (pp["present"] for pp in data["per_product"]
                                                                 if pp["name"] == product["name"]), [])]):
                    st.markdown(f"🔴 {g}")

            with b:
                st.markdown("##### Out: generated content")
                if not agent_content:
                    st.info("No content generated yet for this product — pick a persona and "
                             "click Generate in the Personas tab.")
                else:
                    after_score = readiness_score(completeness_pct, agent_content)
                    st.markdown(f"### {after_score}%")
                    st.caption("readiness — after")
                    comps = score_components(completeness_pct, agent_content)
                    for k, v in comps.items():
                        st.progress(min(v, 1.0), text=f"{LABELS[k]} · {v*100:.0f}%")
                    missing_now = [a for a in data["expected_attrs"]
                                   if a not in next((pp["present"] for pp in data["per_product"]
                                                      if pp["name"] == product["name"]), [])]
                    for g in top_gaps(completeness_pct, agent_content, missing_now):
                        st.markdown(f"🔴 {g}")
                    with st.expander("Generated passage"):
                        st.text(render_passage(agent_content))

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

            options = [
                f"{p['title']} — rating {p['persona_rating_pct']}% "
                f"(covers: {', '.join(p['covered_attributes']) or 'none'})"
                for p in personas
            ]

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
                with st.spinner("Generating user story..."):
                    story = generate_user_story(name, chosen_persona)
                data["selected_persona"] = chosen_persona
                data["user_story"] = story

                with st.spinner(f"Generating agent content for {len(data['members'])} products..."):
                    agent_content_map = {}
                    content_map = {}
                    for product in data["members"]:
                        comps = competitors_for(product, data["members"])
                        ac = generate_agent_content(product, name, chosen_persona, story, comps)
                        agent_content_map[product["name"]] = ac
                        content_map[product["name"]] = render_passage(ac)
                    data["agent_content"] = agent_content_map
                    data["content"] = content_map

                st.success("Story + content generated — see the Dashboard, Generated Content, and Ask tabs.")

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
            for name, data in cd.items():
                if not data["content"]:
                    continue
                st.markdown(f"## {name}")
                if data["user_story"]:
                    st.caption(f"Persona story: {data['user_story']}")

                for product_name, text in data["content"].items():
                    ac = data["agent_content"].get(product_name, {})
                    with st.expander(product_name, expanded=False):
                        st.text_area(
                            "Copy-pastable content",
                            value=text,
                            height=260,
                            key=f"content_{name}_{product_name}",
                        )
                        with st.expander("Structured data (personas, not-for, comparisons)"):
                            st.json({
                                "personas": ac.get("personas", []),
                                "not_for": ac.get("not_for", []),
                                "use_cases": ac.get("use_cases", []),
                                "comparisons": ac.get("comparisons", []),
                                "field_sources": ac.get("field_sources", []),
                            })
                        if ac.get("unsupported_claims"):
                            st.warning("Flagged for human review before publishing:\n\n"
                                       + "\n".join(f"- {u}" for u in ac["unsupported_claims"]))
                st.divider()

# ---------------------------------------------------------------------------
# TAB 5 — Ask (works right after Tab 1 — no persona/content required yet)
# ---------------------------------------------------------------------------
with tab5:
    if not st.session_state["analysis_done"]:
        st.info("Run extraction + analysis in Tab 1 first.")
    else:
        cd = st.session_state["cluster_data"]

        # Every product gets a raw passage (built deterministically, no LLM call,
        # available immediately after Tab 1). Generated passage only exists once
        # a persona has been picked and content generated in Tab 3. This lets you
        # demo the same tab before AND after persona creation.
        all_products = []
        for cname, data in cd.items():
            for product in data["members"]:
                all_products.append({
                    "cluster": cname,
                    "product": product["name"],
                    "raw_passage": build_raw_passage(product),
                    "enriched_passage": data["content"].get(product["name"]),
                })

        any_enriched = any(p["enriched_passage"] for p in all_products)

        st.caption(f"Will check against {len(all_products)} product(s).")
        mode_options = ["Raw catalog content"]
        if any_enriched:
            mode_options += ["Generated content", "Compare both"]
        default_index = mode_options.index("Compare both") if any_enriched else 0
        mode = st.radio("Test against", mode_options, index=default_index, horizontal=True)
        if not any_enriched:
            st.caption("No generated content yet — this will test raw catalog content only. "
                       "Generate content in the Personas tab, then re-run the same query here "
                       "to see the before/after.")

        query = st.text_input("Shopper query",
                               placeholder="e.g. half marathon shoes for humid weather under S$200")

        if st.button("Run", type="primary") and query.strip():
            results = []
            targets = [p for p in all_products if mode != "Generated content" or p["enriched_passage"]]
            bar = st.progress(0.0)
            for i, p in enumerate(targets, 1):
                raw_result = None
                enriched_result = None
                if mode in ("Raw catalog content", "Compare both"):
                    raw_result = ask_confidence(query, p["product"], p["raw_passage"])
                if mode in ("Generated content", "Compare both") and p["enriched_passage"]:
                    enriched_result = ask_confidence(query, p["product"], p["enriched_passage"])
                results.append({
                    "cluster": p["cluster"], "product": p["product"],
                    "raw": raw_result, "enriched": enriched_result,
                })
                bar.progress(i / len(targets))

            def _sort_key(r):
                best = r["enriched"] or r["raw"]
                return -(best["confidence"] if best else 0)
            results.sort(key=_sort_key)
            st.session_state["ask_results"] = results
            st.session_state["ask_mode"] = mode

        def _badge(result):
            if not result:
                return "⚪ —"
            conf = result["confidence"]
            color = "🟢" if conf >= 70 else "🟡" if conf >= 40 else "🔴"
            return f"{color} {conf}%"

        if st.session_state.get("ask_results"):
            st.subheader("Results")
            shown_mode = st.session_state.get("ask_mode", "Raw catalog content")
            for r in st.session_state["ask_results"]:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown(f"**{r['product']}**  ·  _{r['cluster']}_")
                        reason = (r["enriched"] or r["raw"] or {}).get("reason", "")
                        st.caption(reason)
                    with c2:
                        if shown_mode == "Compare both":
                            st.markdown(f"Raw: {_badge(r['raw'])}   →   Generated: {_badge(r['enriched'])}")
                            if r["raw"] and r["enriched"]:
                                delta = r["enriched"]["confidence"] - r["raw"]["confidence"]
                                st.caption(f"{'+' if delta >= 0 else ''}{delta} points")
                        elif shown_mode == "Generated content":
                            st.markdown(_badge(r["enriched"]))
                        else:
                            st.markdown(_badge(r["raw"]))
