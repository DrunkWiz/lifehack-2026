# Brand Enabler

Upload a product catalog → cluster similar products → find attribute & persona gaps →
pick a persona per cluster → generate agent-optimized, structured product content →
see the before/after → ask live queries against it.

## Setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (copy `secrets.toml.example`) with:

```toml
OPENAI_API_KEY = "sk-..."
AGENT_BUDGET_USD = "5.00"   # optional, defaults to 5.00
```

On Streamlit Community Cloud, set the same keys under **App settings → Secrets** instead.

## Run

```bash
streamlit run app.py
```

## How it works

1. **Upload & Extract** — upload a PDF, CSV, XLSX, or JSON catalog.
   - CSV/XLSX parsed directly with pandas.
   - JSON: auto-detects Shopify's nested `products.json` shape (variants, options,
     tags, body_html) or a flat generic product list.
   - PDF: each page's selectable text is extracted first; if a page has too little
     text (scanned page / image-heavy brochure), that page is rendered as an image
     and read by a vision-capable model instead.
   - Extracted products are shown in an editable table before anything is analyzed.

2. **Clustering** — products are grouped into similarity clusters (batched in chunks
   of 20 per LLM call; larger catalogs get a follow-up merge pass so cluster labels
   stay consistent across batches).

3. **Gap analysis (per cluster)** — the model infers the attributes a buyer/agent
   would expect for that category, scores what % are present (**Attribute
   Completeness**), and suggests 2-3 candidate personas, each rated on how many of
   its needed attributes the data actually supports.

4. **Personas tab** — pick one persona per cluster. This generates a short user
   story, then — **one structured call per product** — a richer content bundle:
   - `personas` (multiple, **must include at least one "poor fit"**)
   - `not_for` (negative information — required, minimum 2 entries)
   - `use_cases` grounded in real spec values
   - `comparisons` against real sibling products in the same cluster, each with a
     mandatory tradeoff
   - `narrative` (the pitch, best-for line, FAQ)
   - `field_sources` (lightweight provenance: catalog_spec vs. inferred) +
     `unsupported_claims` (flagged for human review before publishing)

   The copy-pastable text is then **rendered from these fields with plain string
   formatting — no further model call** — so nothing new can be invented at that step.

5. **Dashboard** — headline `st.metric` for overall catalog readiness, a per-cluster
   breakdown, and a **Before/After** section: pick one product, see the raw catalog
   row and its score on the left, the generated content and its score (broken into
   the 5 weighted components) on the right.

   **Readiness Score** — deterministic, 5 weighted components:
   ```
   attribute_completeness   25%   expected attributes present in the catalog data
   persona_coverage         20%   distinct personas, penalized if none is "poor" fit
   not_for_coverage         15%   negative information present
   comparative_context      15%   comparisons that state a real tradeoff
   claim_grounding          25%   share of fields traced to catalog data vs. inferred
   ```
   Before a persona is generated, only `attribute_completeness` contributes — that's
   what gives the Before/After view its "before" number.

6. **Generated Content tab** — final copy-pastable text per product, plus an
   expander with the underlying structured data (personas, not_for, comparisons,
   provenance) and a review flag for anything marked as an unsupported claim.

7. **Ask tab** — works right after Tab 1 (no persona/content required), which
   makes it the live-demo mechanism: type a natural-language shopper query, run
   it, and see confidence scores against every product's **raw catalog content**.
   Then go generate content for a persona in the Personas tab, come back, and
   re-run the *same* query — the tab now offers "Generated content" and
   "Compare both" modes, showing the raw and generated confidence side by side
   with the point delta between them. One call per product per mode asks
   whether the model would recommend it based on that content alone, returning
   a confidence score (0-100) and a one-line reason. Results are ranked, with a
   color-coded confidence badge (🟢 ≥70, 🟡 40-69, 🔴 <40, ⚪ not tested).

## Cost control

Every LLM call (JSON, text, or vision) is cached to disk under `.cache/completions/`,
keyed by a hash of the exact prompt + model + temperature — re-running the same
product through the pipeline again costs nothing. A running spend ledger
(`.cache/spend.json`) tracks cost, and calls are refused once total spend crosses
`AGENT_BUDGET_USD` (default $5). Delete `.cache/` to reset both.

## Notes

- Model used: `gpt-4o-mini` (vision + text), configurable in `llm_utils.py`.
- No login, no user-facing persistence — catalog/analysis state lives in
  `st.session_state` for the session; only the LLM cache/spend ledger persists on disk.
- Uploaded catalog content is sent to OpenAI's API for processing (disclosed in-app).
- `test_personas.py` runs the whole pipeline on a small hardcoded cluster outside
  the UI — useful for checking prompt/logic changes quickly.
