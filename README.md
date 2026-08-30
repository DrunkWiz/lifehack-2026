# AgentShopper — AI-Ready Catalogue Assessment & Content Generation

Upload a product catalog → cluster similar products → measure attribute & persona gaps → generate agent-optimized product content that AI shopping assistants can understand and recommend.

**Live Demo:** https://agentshopper.streamlit.app/

---

## Project Overview

**Problem:** AI shopping agents fail silently on incomplete product catalogues. Catalogues built for human shoppers—who can infer context and tolerate ambiguity—break agents immediately.

**Solution:** AgentShopper audits product catalogues and assigns an objective readiness score. The system extracts data from any format (CSV, JSON, PDF, Shopify), infers optimal schemas through two-pass inference, evaluates fit against real personas using deterministic Python checks, and generates gap reports showing exactly what to fix.

**Challenge:** LifeHack 2026 — Rezolve AI

---

## Team

- **Nathan Quek** — Engineering, AI Reasoning, Algo Checker
- **Vibu Vignesh** — Full-Stack Development, UI/UX Engineering, Architecture
- **Mingyu** —  Data Pipeline, Product Strategy

---

## Quick Start for Judges

No installation required. Access the live app directly:

**https://agentshopper.streamlit.app/**

### To Evaluate:

1. **Upload a catalogue** in the **Upload & Extract** tab
   - Use the sample CSVs provided (running shoes or facial skincare)
   - Or upload your own CSV/XLSX/JSON/PDF
2. **Review extracted data** — table shows what was parsed
3. **Watch the analysis** run across 7 steps:
   - Clustering → Schema inference → Completeness → Personas → Story → Content generation → Readiness scoring
4. **Explore results** in the Dashboard:
   - Overall catalogue readiness metric
   - Per-cluster breakdown (attributes, personas, gaps)
5. **Test live queries** in the **Ask like a Shopper** tab:
   - *"I'm training for a half marathon in humid weather, under SGD 200"*
   - *"Best anti-aging serum for sensitive skin"*
6. **Review generated content** in the **Generated Content** tab

---

## How It Works

### 1. Upload & Extract
- Accept CSV, XLSX, JSON, PDF, or Shopify exports
- CSV/XLSX parsed directly
- PDFs: text extraction first; vision fallback for image-heavy pages
- Editable table for manual corrections before analysis
- Blank fields treated as content gaps, not errors

### 2. Clustering
- Group products by semantic similarity (batched processing)
- Per-batch clustering merged to maintain label consistency
- Category-agnostic (works on any product type)

### 3. Gap Analysis (Per Cluster)
- **Attribute Completeness:** Infer expected attributes for category, score % present (0–100%)
- **Persona Coverage:** Suggest 2–3 candidate personas; score fit based on required attributes
- **Not-For Coverage:** Identify personas the product explicitly doesn't suit
- **Comparative Context:** Detect positioning claims vs alternatives
- **Claim Grounding:** Verify assertions against normalized, verified attributes

### 4. Readiness Score

```
Readiness = 25% attribute_completeness
          + 20% persona_coverage
          + 15% not_for_coverage
          + 15% comparative_context
          + 25% claim_grounding
```

Product *fit* (does this persona actually want this product?) is shown separately because it's a merchandising outcome, not a content gap.

### 5. Personas Tab
- Pick one persona per cluster
- Auto-generate user story: *"As a [persona], I need [attribute] so that [outcome]"*
- Generate agent-optimized product content seeded by story

### 6. Dashboard
- Overall catalogue readiness metric
- Per-cluster breakdown (missing attributes, persona, score)
- Trend visualization

### 7. Generated Content
- Copy-pastable product descriptions
- Comparisons verified against peer products
- Unsupported claims quarantined for review
- Every claim cites verified source attribute
- Includes derived insights, FAQ angles, use cases, explicit exclusions

---

## Key Features

- **Multi-Format Extraction:** CSV, XLSX, JSON, PDF, Shopify—automatically parsed and normalised with verbatim verification
- **Two-Pass Schema Inference:** Identify candidate attributes; verify which products actually match
- **Deterministic Fit Scoring:** Convert personas to machine-checkable criteria, evaluate with pure Python (no subjective LLM ratings)
- **Dual Indexing:** Raw catalogue + agent-optimised summaries for before/after comparison
- **Verbatim Verification:** Every extracted value must exist word-for-word in source text—no hallucination
- **Interactive Dashboard:** Streamlit UI with 5 analysis tabs: Extract, Ask, Dashboard, Personas, Content
- **Export Formats:** CSV (merchandisers), JSON (developers), schema.org JSON-LD (embed in product pages)

---

## Sample Data

Two demo catalogues are available within the app:
- **Running Shoes (20 products):** Ranging from fully-specified to minimal data
- **Facial Skincare (20 products):** Different category to test generalization

---

## Technology Stack

- **Backend:** Python 3.9+
- **UI Framework:** Streamlit (interactive dashboard, deployed to Streamlit Community Cloud)
- **LLM & Reasoning:** OpenAI API (`gpt-4o-mini` for text + vision)
- **Data Processing:** Pandas, scikit-learn (clustering, embeddings)
- **Validation:** Custom Python evaluation (deterministic fit scoring)
- **Caching:** Disk cache (SHA256-keyed LLM results)
- **Export:** JSON, CSV, schema.org JSON-LD

---

## Third-Party Acknowledgments

### Libraries & Frameworks
- **Streamlit** — Interactive web UI framework
- **Pandas** — Data manipulation and analysis
- **scikit-learn** — Clustering algorithms, similarity metrics
- **NumPy** — Numerical computation
- **PyPDF2** — PDF text extraction
- **OpenPyXL** — XLSX parsing

### APIs & Services
- **OpenAI API** — GPT-4O Mini (text generation, vision understanding, embeddings)
  - Used for: schema inference, persona generation, content generation, embeddings
  - Cost management: disk caching with SHA256 keying, budget guards
- **Streamlit Community Cloud** — Application hosting and deployment

### Datasets
- **Shopify Product CSV Examples** — Running shoes and facial skincare sample catalogues (20 products each)
  - Intentional data quality gradient (FULL/TERSE/EMPTY specs) for testing
  - Used to demonstrate category-agnostic clustering and gap analysis

---

## Performance Notes

- **Latency:** ~1-3 minutes for 20-product catalogue (includes LLM calls)
- **Large Catalogs:** Serial processing; 200+ products will be noticeably slower
- **LLM API:** All catalog content is sent to OpenAI for processing (disclosed in-app)
- **Session-Based:** Data lives in Streamlit session state; reloading clears analysis

---

## Known Limitations

- **Serial Processing:** Extraction, clustering, schema, personas, content all run sequentially
- **No Vector DB:** Embeddings stored in NumPy arrays (not suitable for 10k+ products without refactoring)
- **Session-Based:** No persistent storage between sessions
- **Cluster Naming Collision:** Duplicate cluster names in multi-batch processing can cause data loss (planned fix: switch to embeddings + agglomerative clustering)

---

## License

Built for LifeHack 2026 — Rezolve AI Challenge.

---

## Questions?

For technical questions, open an issue on GitHub or refer to the internal project documentation.
