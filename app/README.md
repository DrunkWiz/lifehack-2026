# Agent Readiness Copilot

Upload a product catalog → cluster similar products → find attribute & persona gaps →
pick a persona per cluster → generate agent-optimized, copy-pastable product content.

## Setup

```bash
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` (copy `secrets.toml.example`) with:

```toml
OPENAI_API_KEY = "sk-..."
```

On Streamlit Community Cloud, set the same key/value under **App settings → Secrets** instead.

## Run

```bash
streamlit run app.py
```

## How it works

1. **Upload & Extract** — upload a PDF, CSV, or XLSX catalog.
   - CSV/XLSX are parsed directly.
   - PDFs: each page's selectable text is extracted first; if a page has too little
     text (scanned page / image-heavy brochure), that page is rendered as an image
     and read by a vision-capable model instead.
   - Extracted products are shown in an editable table — fix anything before running
     analysis. Blank fields are treated as gaps, not extraction errors.

2. **Clustering** — products are grouped into similarity clusters (batched in chunks
   of 20 products per LLM call; if the catalog is larger, clusters from each batch
   are merged by a follow-up pass so labels stay consistent).

3. **Gap analysis (per cluster)**
   - The model infers the attributes a buyer/agent would expect for that category,
     then scores what % are actually present → **Attribute Completeness**.
   - 2-3 candidate personas (natural shopping-intent phrasing, not marketing personas)
     are suggested per cluster. Each persona lists which attributes it depends on, and
     is scored on how many of those are actually present in the data → **Persona Rating**.

4. **Personas tab** — pick one persona per cluster. This generates a short user story
   ("As a ___, I need ___ so that ___") and then agent-optimized content for every
   product in that cluster, seeded by the story.

5. **Dashboard** — headline `st.metric` for overall catalog readiness, plus a
   per-cluster breakdown (missing attributes, selected persona, readiness score).

   **Readiness Score = 30% × Attribute Completeness + 70% × Persona Rating**
   (of the selected persona; 0% before a persona is chosen).

6. **Generated Content tab** — final copy-pastable text per product: a semantic
   passage, a "Best for" line, and a short FAQ — grounded only in real attributes,
   nothing fabricated.

## Notes

- Model used: `gpt-4o-mini` (vision + text), configurable in `llm_utils.py`.
- No login, no persistence — everything lives in `st.session_state` for the session.
- Uploaded catalog content is sent to OpenAI's API for processing (disclosed in-app).
