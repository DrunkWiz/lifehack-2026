import streamlit as st
import pandas as pd

from extraction import load_catalog_file
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🧭 Agent Readiness Copilot")
st.caption("Upload a product catalog → find AI-recommendation gaps → generate persona-driven, agent-optimized content.")

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Upload & Extract", "2️⃣ Dashboard", "3️⃣ Personas", "4️⃣ Generated Content"])

# ---------------------------------------------------------------------------
# TAB 1 — Upload & Extract (+ confirm/edit merged in here)
# ---------------------------------------------------------------------------
with tab1:
    st.info("Uploaded content is sent to OpenAI's API for processing.", icon="ℹ️")

    uploaded_file = st.file_uploader("Upload catalog (PDF, CSV, XLSX, or JSON)", type=["pdf", "csv", "xlsx", "xls", "json"])

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
                # reset downstream state since catalog changed
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
                "specs": ", ".join(f"{k}: {v}" for k, v in p.get("specs", {}).items()),
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
                        "content": {},
                    }
            st.session_state["cluster_data"] = cluster_data
            st.session_state["analysis_done"] = True
            st.success("Analysis complete — see the Dashboard and Personas tabs.")

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
        st.subheader("Per-cluster breakdown")

        for name, data in cd.items():
            persona = data["selected_persona"]
            persona_rating = persona["persona_rating_pct"] if persona else None
            score = readiness_score(data["avg_completeness"], persona_rating)

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
                if missing:
                    st.markdown(f"**Missing attributes:** {', '.join(missing)}")
                else:
                    st.markdown("**Missing attributes:** none 🎉")

                if persona:
                    st.markdown(
                        f"**Selected persona:** {persona['title']} "
                        f"(persona rating: {persona['persona_rating_pct']}%)"
                    )
                else:
                    st.markdown("**Selected persona:** none yet — pick one in the Personas tab.")

                with st.expander("Per-product attribute detail"):
                    for p in data["per_product"]:
                        st.write(
                            f"- **{p['name']}** — {p['completeness_pct']}% "
                            f"(missing: {', '.join(p['missing']) if p['missing'] else 'none'})"
                        )

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

                with st.spinner(f"Generating content for {len(data['members'])} products..."):
                    content_map = {}
                    for product in data["members"]:
                        content_map[product["name"]] = generate_product_content(
                            product, name, chosen_persona, story
                        )
                    data["content"] = content_map

                st.success("Story + content generated — see the Generated Content tab.")

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
                    with st.expander(product_name, expanded=False):
                        st.text_area(
                            "Copy-pastable content",
                            value=text,
                            height=220,
                            key=f"content_{name}_{product_name}",
                        )
                st.divider()
