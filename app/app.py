import hashlib
import json
from pathlib import Path

import streamlit as st
import pandas as pd

from extraction import load_catalog_file
from normalization import normalize_cluster, infer_applicability
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
    AGENT_CONTENT_SCHEMA_VERSION,
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
    suggest_shopper_queries,
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
    "query_input": "I'm training for a half marathon in Singapore's humid weather and need lightweight shoes under S$200.",
    "index_signature": None,   # invalidates the cached indexes when the catalog/content changes
    "raw_index": None,
    "optimized_index": None,
    "query_result": None,
    "suggested_queries": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "api_usage_baseline" not in st.session_state:
    usage = get_spend_summary()
    st.session_state["api_usage_baseline"] = {
        key: usage.get(key, 0)
        for key in ("calls", "cached", "usd", "input_tokens", "output_tokens")
    }

st.title("🛒 AgentShopper")
st.caption("Upload a product catalog → find AI-recommendation gaps → generate persona-driven, agent-optimized content.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_CATALOGS = {
    "Running shoes": PROJECT_ROOT / "data" / "shopify_running_shoes.csv",
    "Facial skincare": PROJECT_ROOT / "data" / "shopify_facial_skincare.csv",
}


def _source_product_id(product: dict) -> str:
    candidates = {str(k).strip().lower(): v for k, v in (product.get("specs") or {}).items()}
    for key in ("variant sku", "sku", "handle", "product id", "id"):
        value = str(candidates.get(key) or "").strip()
        if value:
            return value
    return ""


def _ensure_product_ids(products: list[dict]) -> None:
    """Attach a session-stable, unique identifier without relying on display names."""
    used = set()
    for position, product in enumerate(products):
        existing = str(product.get("product_id") or _source_product_id(product)).strip()
        base = existing or hashlib.sha256(
            f"{position}|{product.get('name')}|{product.get('description')}".encode("utf-8")
        ).hexdigest()[:16]
        product_id = base
        suffix = 2
        while product_id in used:
            product_id = f"{base}-{suffix}"
            suffix += 1
        product["product_id"] = product_id
        used.add(product_id)


def _migrate_session_ids() -> None:
    """Keep an active pre-upgrade Streamlit session usable after switching map keys."""
    products = st.session_state.get("products") or []
    _ensure_product_ids(products)
    for data in (st.session_state.get("cluster_data") or {}).values():
        members = data.get("members") or []
        _ensure_product_ids(members)
        name_to_ids = {}
        for product in members:
            name_to_ids.setdefault(str(product.get("name") or "Unnamed product"), []).append(
                str(product["product_id"])
            )
        for field in ("content", "agent_content"):
            old = data.get(field) or {}
            if old and not all(key in {str(p["product_id"]) for p in members} for key in old):
                migrated = {}
                for key, value in old.items():
                    ids = name_to_ids.get(str(key), [])
                    if len(ids) == 1:
                        migrated[ids[0]] = value
                data[field] = migrated
        per_product = data.get("per_product") or []
        for position, record in enumerate(per_product):
            if not record.get("product_id") and position < len(members):
                record["product_id"] = str(members[position]["product_id"])


_migrate_session_ids()


def _set_extracted_products(products: list[dict]) -> None:
    """Install a newly loaded catalog and clear results belonging to the previous one."""
    _ensure_product_ids(products)
    st.session_state["products"] = products
    st.session_state["clusters"] = None
    st.session_state["cluster_data"] = {}
    st.session_state["analysis_done"] = False
    st.session_state["index_signature"] = None
    st.session_state["raw_index"] = None
    st.session_state["optimized_index"] = None
    st.session_state["query_result"] = None
    st.session_state["suggested_queries"] = None

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

    st.subheader("Try a demo catalog")
    st.caption("Load a prepared Shopify export, review the extracted products, then run the same analysis used for uploads.")
    demo_cols = st.columns(2)
    for col, (demo_name, demo_path) in zip(demo_cols, DEMO_CATALOGS.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"**{demo_name}**")
                st.caption("20 products · Shopify CSV")
                if st.button(f"Load {demo_name} demo", key=f"demo_{demo_name}", width="stretch"):
                    log = RunLog(f"Loading {demo_name} demo")
                    try:
                        with demo_path.open("rb") as demo_file:
                            products = load_catalog_file(demo_file)
                        _set_extracted_products(products)
                        log.done(f"Loaded {len(products)} demo products")
                        st.success(f"{demo_name} demo loaded. Review the products below, then run analysis.")
                    except Exception as e:
                        log.fail(str(e))

    st.divider()
    st.subheader("Or upload your own catalog")

    uploaded_file = st.file_uploader("Upload catalog (PDF, CSV, XLSX, or JSON)",
                                      type=["pdf", "csv", "xlsx", "xls", "json"])

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
            _set_extracted_products(products)
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
                "product_id": p.get("product_id", ""),
                "name": p.get("name", ""),
                "price": p.get("price", ""),
                "description": p.get("description", ""),
                # JSON, not "k: v, k: v" - a comma inside any value (an ingredient list,
                # "258g (US M9), wide") used to be split into garbage on the way back.
                "specs": json.dumps(p.get("specs", {}), ensure_ascii=False),
            }
            for p in st.session_state["products"]
        ])
        edited_df = st.data_editor(
            df, num_rows="dynamic", width="stretch", key="edit_products",
            column_config={"product_id": None},
        )

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
                    "product_id": clean_str(row.get("product_id")) or None,
                    "name": clean_str(row["name"]) or "Unnamed product",
                    "price": clean_str(row["price"]) or None,
                    "description": clean_str(row["description"]),
                    "specs": specs,
                })
            _ensure_product_ids(rebuilt)
            st.session_state["products"] = rebuilt
            st.session_state["suggested_queries"] = None

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
                    if expected_attrs and not all("applicable_attributes" in p for p in members):
                        try:
                            applicability = infer_applicability(name, expected_attrs, members)
                        except Exception:
                            applicability = []
                        for index, product in enumerate(members):
                            product["applicable_attributes"] = (
                                applicability[index] if applicability else list(expected_attrs)
                            )
                        norm_stats["applicability_inferred"] = bool(applicability)
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
                        "agent_content": {},   # product_id -> structured dict
                        "content": {},         # product_id -> rendered copy-pastable text
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

DEMO_QUERY_EXAMPLES = [
    ("👟 Running Shoes demo", "I'm training for a half marathon in Singapore's humid weather and need lightweight shoes under S$200."),
    ("🧴 Facial Skincare demo", "Find me a sustainable skincare routine for oily skin that takes less than 5 minutes every morning."),
]


def _set_query(text: str):
    st.session_state["query_input"] = text


def _content_signature(cluster_data: dict) -> str:
    """Changes whenever the catalog or the generated content changes, so the cached
    embeddings are rebuilt instead of silently going stale."""
    # Bump when index text construction changes so active sessions do not reuse stale vectors.
    parts = ["index-v6-evidence-backed-derived-insights"]
    for cluster_name, data in cluster_data.items():
        for product in data.get("members", []):
            parts.append(f"{cluster_name}|{product.get('product_id')}|{len(str(product.get('specs') or {}))}")
        for product_id, text in (data.get("content") or {}).items():
            parts.append(f"{cluster_name}|{product_id}|gen{len(text)}")
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def _build_indexes(cluster_data: dict):
    """Both indexes cover EXACTLY the same products — those that have generated content —
    so the only thing that differs between Before and After is the content itself.

    Including un-optimized products in the raw index only would rig the comparison: the
    Before side would search the whole catalog while the After side searched a subset,
    and the After side would be forced to return whatever it had regardless of fit."""
    raw_entries, optimized_entries = [], []
    indexed_ids = set()
    for cluster_name, data in cluster_data.items():
        content_map = data.get("content") or {}
        for product in data.get("members", []):
            name = str(product.get("name") or "Unnamed product")
            product_id = str(product.get("product_id") or "")
            generated = content_map.get(product_id)
            if not generated or product_id in indexed_ids:
                continue  # excluded from BOTH sides, never just one
            indexed_ids.add(product_id)
            key = product_id
            facts = dict(product.get("specs_normalized") or {})
            facts["price"] = product.get("price")
            facts["category"] = cluster_name
            raw_entries.append({
                "key": key, "name": name, "cluster": cluster_name,
                "text": product_to_raw_text(product),
                "attributes": facts,
            })
            optimized_entries.append({
                "key": key, "name": name, "cluster": cluster_name,
                "text": product_to_optimized_text(product, generated),
                "attributes": facts,
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
    pending = [
        (name, data) for name, data in cluster_data.items()
        if any(
            (data.get("agent_content") or {}).get(str(product.get("product_id")), {}).get(
                "schema_version"
            ) != AGENT_CONTENT_SCHEMA_VERSION
            for product in data.get("members", [])
        )
    ]
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
        agent_content = dict(data.get("agent_content") or {})
        for product in data.get("members", []):
            product_id = str(product.get("product_id") or "")
            if agent_content.get(product_id, {}).get("schema_version") == AGENT_CONTENT_SCHEMA_VERSION:
                continue
            structured = generate_agent_content(
                product,
                cluster_name,
                persona,
                data["user_story"],
                competitors_for(product, data.get("members", [])),
            )
            agent_content[product_id] = structured
            content[product_id] = render_passage(structured)
        data["agent_content"] = agent_content
        data["content"] = content


def _render_hits(hits: list, empty_message: str):
    def safe_markdown(value) -> str:
        # Streamlit Markdown treats paired dollar signs as a LaTeX expression.
        return str(value).replace("$", r"\$")

    if not hits:
        st.caption(empty_message)
        return
    for hit in hits:
        marker = "✅" if hit.get("recommend") else "⚠️"
        st.markdown(f"**{hit['final_rank']}. {hit['name']}** {marker}")
        st.caption(f"{hit.get('cluster','')} · similarity {hit['score']}")
        if hit.get("reason"):
            st.write(safe_markdown(hit["reason"]))
        if hit.get("cited_attributes"):
            st.markdown("**Cited:** " + ", ".join(f"`{a}`" for a in hit["cited_attributes"]))
        checks = hit.get("constraint_checks") or []
        if checks:
            icons = {"pass": "✅", "fail": "❌", "unknown": "❓"}
            st.markdown("**Hard-constraint checks:** " + " · ".join(
                f"{icons.get(check.get('status'), '❓')} {safe_markdown(check.get('requirement', ''))}"
                for check in checks
            ))
        if hit.get("failed_requirements"):
            st.markdown("**Does not meet:** " + ", ".join(
                safe_markdown(item) for item in hit["failed_requirements"]
            ))
        if hit.get("unanswered"):
            st.markdown("**Content can't answer:** " + ", ".join(
                safe_markdown(item) for item in hit["unanswered"]
            ))
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

        st.markdown("#### Questions for the sample data")
        st.caption("These fixed examples are designed for the two demo catalogs available in Upload & Extract (Tab 1).")
        demo_query_cols = st.columns(2)
        for i, (label, example) in enumerate(DEMO_QUERY_EXAMPLES):
            with demo_query_cols[i]:
                with st.container(border=True):
                    st.caption(label)
                    st.button(
                        example,
                        key=f"demo_query_{i}",
                        on_click=_set_query,
                        args=(example,),
                        width="stretch",
                    )

        if st.session_state.get("suggested_queries") is None:
            try:
                with st.spinner("Creating questions for this catalog..."):
                    st.session_state["suggested_queries"] = suggest_shopper_queries(cd)
            except Exception:
                st.session_state["suggested_queries"] = []

        fixed_text = {example for _, example in DEMO_QUERY_EXAMPLES}
        smart_queries = [query for query in (st.session_state.get("suggested_queries") or [])
                         if query not in fixed_text]
        if smart_queries:
            st.markdown("#### Suggested from the current catalog")
            smart_cols = st.columns(len(smart_queries))
            for i, (col, example) in enumerate(zip(smart_cols, smart_queries)):
                with col:
                    st.button(
                        example,
                        key=f"smart_query_{i}",
                        on_click=_set_query,
                        args=(example,),
                        width="stretch",
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

            intent = result.get("intent") or {}
            task = intent.get("task") or {}
            contexts = intent.get("context") or []
            constraints = intent.get("constraints") or []
            hard_constraints = [c for c in constraints if c.get("strength") == "hard"]
            preferences = [c for c in constraints if c.get("strength") != "hard"]
            derived_needs = intent.get("derived_needs") or []

            def _intent_text(items, primary_key, fallback="Not stated"):
                values = [str(item.get(primary_key) or item.get("source_text") or "").strip()
                          for item in items]
                unique_values = list(dict.fromkeys(value for value in values if value))
                return " · ".join(unique_values) or fallback

            st.markdown("#### Query understanding")
            intent_cols = st.columns(5)
            summaries = [
                ("Task", task.get("source_text") or task.get("goal") or "Not identified"),
                ("Context", _intent_text(contexts, "value")),
                ("Hard constraints", _intent_text(hard_constraints, "requirement", "None")),
                ("Preferences", _intent_text(preferences, "requirement", "None")),
                ("Derived needs", _intent_text(derived_needs, "need", "None")),
            ]
            for col, (label, value) in zip(intent_cols, summaries):
                with col:
                    with st.container(border=True):
                        st.caption(label)
                        st.write(str(value).replace("$", r"\$"))

            with st.expander("View full interpretation details"):
                st.json(intent)

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

        def _product_completeness_pct(data, product_id):
            for p in data["per_product"]:
                if p.get("product_id") == product_id:
                    return p["completeness_pct"]
            return data["avg_completeness"]

        def _cluster_score(data):
            scores = []
            for p in data["per_product"]:
                agent_content = data["agent_content"].get(p.get("product_id"))
                scores.append(readiness_score(p["completeness_pct"], agent_content))
            return round(sum(scores) / len(scores), 1) if scores else 0.0

        cluster_scores = [_cluster_score(data) for data in cd.values()]
        overall = round(sum(cluster_scores) / len(cluster_scores), 1) if cluster_scores else 0.0

        st.metric("Overall Catalog Readiness Score", f"{overall}%",
                   help="25% attribute completeness, 20% persona coverage, 15% not-for coverage, "
                        "15% verified comparison context, and 25% claim grounding; averaged across clusters.")

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
                f"{worst['missing_count']}/{worst['applicable_products']} applicable products "
                f"in {worst['cluster']}."
            )
            attr_rows = gaps.attribute_requests(cd)
            prod_rows = gaps.product_requests(cd)
            st.dataframe(
                pd.DataFrame(attr_rows)[["cluster", "attribute", "applicable_products", "missing_count",
                                          "missing_pct", "needed_because"]],
                width="stretch", hide_index=True,
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
            score = _cluster_score(data)

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
                    if ns.get("applicability_inferred"):
                        bits.append("attribute relevance mapped per product")
                    st.caption("Knowledge layer: " + " · ".join(bits))
                else:
                    st.caption("Knowledge layer: not applied - using raw specs and inferred attributes.")

                missing = [attr for attr, count in data["missing_counts"].items() if count > 0]
                st.markdown(f"**Missing attributes:** {', '.join(missing) if missing else 'none 🎉'}")

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
                        if p.get("not_applicable"):
                            st.caption("   not applicable: " + ", ".join(p["not_applicable"]))
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
                    agent_content_map = {}
                    total = len(data["members"])
                    for i, product in enumerate(data["members"]):
                        pname = str(product.get("name", "Unnamed product"))
                        product_id = str(product.get("product_id") or "")
                        grounded = ("verified attributes" if product.get("specs_normalized")
                                    else "no verified attributes; copy will be skipped")
                        log.step(0.1 + 0.9 * (i / max(total, 1)),
                                 f"{i + 1}/{total} — {pname} (from {grounded})")
                        structured = generate_agent_content(
                            product,
                            name,
                            chosen_persona,
                            story,
                            competitors_for(product, data["members"]),
                        )
                        agent_content_map[product_id] = structured
                        content_map[product_id] = render_passage(structured)
                    data["agent_content"] = agent_content_map
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

                products_by_id = {str(p.get("product_id")): p for p in data["members"]}
                for product_id, text in data["content"].items():
                    product = products_by_id.get(product_id, {})
                    product_name = str(product.get("name") or "Unnamed product")
                    ac = data["agent_content"].get(product_id, {})
                    with st.expander(product_name, expanded=False):
                        fact_col, insight_col = st.columns(2)
                        with fact_col:
                            st.markdown("**Verified facts**")
                            st.json(product.get("specs_normalized") or {})
                        with insight_col:
                            st.markdown("**Evidence-backed derived insights**")
                            insights = ac.get("derived_insights") or []
                            if insights:
                                for insight in insights:
                                    st.write(f"• {insight.get('claim', '')}")
                                    st.caption(
                                        "Evidence: " + ", ".join(
                                            f"{e.get('attribute')}: {e.get('value')}"
                                            for e in insight.get("evidence", [])
                                        )
                                    )
                            else:
                                st.caption("No supported derivations generated.")
                        st.text_area(
                            "Copy-pastable content",
                            value=text,
                            height=420,
                            key=f"content_{name}_{product_id}",
                        )
                        with st.expander("Structured knowledge layer"):
                            st.json({
                                "personas": ac.get("personas", []),
                                "derived_insights": ac.get("derived_insights", []),
                                "query_angles": ac.get("query_angles", []),
                                "not_for": ac.get("not_for", []),
                                "use_cases": ac.get("use_cases", []),
                                "comparisons": ac.get("comparisons", []),
                                "field_sources": ac.get("field_sources", []),
                            })
                        if ac.get("unsupported_claims"):
                            st.warning("Flagged for human review before publishing:\n\n"
                                       + "\n".join(f"- {u}" for u in ac["unsupported_claims"]))
                st.divider()

# Render last so calls made during this run are reflected immediately.
usage = get_spend_summary()
baseline = st.session_state["api_usage_baseline"]
session_calls = max(0, usage.get("calls", 0) - baseline.get("calls", 0))
session_cached = max(0, usage.get("cached", 0) - baseline.get("cached", 0))
session_cost = max(0.0, usage.get("usd", 0.0) - baseline.get("usd", 0.0))
session_input = max(0, usage.get("input_tokens", 0) - baseline.get("input_tokens", 0))
session_output = max(0, usage.get("output_tokens", 0) - baseline.get("output_tokens", 0))

with st.sidebar:
    st.header("API usage")
    if usage.get("api_key_configured"):
        st.success("API key configured", icon="🔐")
    else:
        st.error("API key not configured", icon="🔒")
    st.caption("The key itself is never displayed.")

    c1, c2 = st.columns(2)
    c1.metric("Live calls", session_calls)
    c2.metric("Cache hits", session_cached)
    st.metric("Estimated session cost", f"US${session_cost:.4f}")
    st.caption(f"Tokens this session: {session_input:,} input · {session_output:,} output")

