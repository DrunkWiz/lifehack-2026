# 🛒 AgentShopper — AI-Ready Catalogue Assessment & Content Generation

**Turn an ordinary product catalog into content that AI shopping assistants can actually understand, trust, and recommend — and then prove it worked.**

### 🔗 Live demo: **https://agentshopper.streamlit.app/**

**Challenge:** LifeHack 2026 — Rezolve AI

---

## Team

- **Nathan Quek** — Engineering, AI Reasoning, Algo Checker
- **Vibu Vignesh** — Full-Stack Development, UI/UX Engineering, Architecture
- **Mingyu** — Data Pipeline, Product Strategy

---

## The problem, in plain words

People are starting to shop by *asking* instead of *searching*.

The old way: someone types **"running shoes size 10"** into a search box.

The new way: someone asks an AI assistant **"I'm running a half marathon in Singapore's humid weather and need lightweight shoes under S$200."**

To answer that second question, an AI needs to know things a normal product page never says out loud:

- Will these shoes cope with heat and humidity?
- Are they built for that kind of distance?
- How do they compare to the other shoes on this site?
- Do they fit the budget?

Most catalogs only contain a name, a price, and a spec list. That's not enough for a machine to reason with — so the product simply never gets recommended. AI shopping agents fail *silently* on catalogues built for humans, who can infer context and tolerate ambiguity. **AgentShopper closes that gap.**

---

## What it does, in five steps

| Step | What happens |
|---|---|
| **1. Upload** | Drop in a catalog file (or click a demo). The app pulls out every product. |
| **2. Organise** | Products are grouped into categories, and their messy spec text is turned into clean, comparable facts. |
| **3. Diagnose** | You get an objective readiness score and a list of exactly what information is missing — and which products it's missing from. |
| **4. Write** | For each product it writes agent-ready content: who it suits, who it *doesn't*, real comparisons, and answers to shopper questions. Every claim has to cite a real fact from your data. |
| **5. Prove** | Type a real shopper question. The app searches your catalog **twice** — once using your original content, once using the new content — and shows the two result lists side by side. |

That last step is the important one. The app doesn't just *claim* the content is better; it shows you a product moving from "never showed up at all" to "#1 recommendation".

---

## Quick start for judges

**No installation required.** Open **https://agentshopper.streamlit.app/** and:

1. **Load a catalogue** in the **Upload & Extract** tab — click one of the two built-in demos (running shoes or facial skincare), or upload your own CSV / XLSX / JSON / PDF.
2. **Review the extracted data.** Everything appears in an editable table before analysis runs. Fix anything you like.
3. **Hit "Confirm & run"** and watch the live progress log work through clustering → schema inference → attribute cleaning → completeness → personas → fit criteria → content generation → scoring.
4. **Read the Dashboard** — overall readiness score, per-cluster breakdown, and a downloadable list of exactly what data is missing.
5. **Test it live** in the **Ask like a Shopper** tab. Type any question you like — nothing is hard-coded. Two built-in examples:
   - *"I'm training for a half marathon in Singapore's humid weather and need lightweight shoes under S$200."*
   - *"Find me a sustainable skincare routine for oily skin that takes less than 5 minutes every morning."*

   The tab also generates three fresh questions from whatever catalogue you loaded.
6. **Inspect the output** in the **Generated Content** tab, and export it as CSV, JSON, or schema.org JSON-LD.

> **Try to break it.** Ask about something the catalogue can't support. The app is built to *say so* rather than bluff — that behaviour is the point, not a failure.

---

## Running it locally

**You need:** Python **3.10 or newer** (the code uses `X | None` type syntax), and an OpenAI API key.

```bash
cd app
pip install -r requirements.txt
```

Put your key in `app/.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
AGENT_BUDGET_USD = 5.00   # optional spending cap, defaults to $5
```

Then run it:

```bash
streamlit run app.py
```

On Streamlit Community Cloud, skip the file and paste the same key under **App settings → Secrets**.

---

## How it works, feature by feature

### Tab 1 — Upload & Extract

**Accepts almost any catalog file:**

- **CSV / Excel** (`.csv`, `.xlsx`, `.xls`) — read directly.
- **JSON** — handles both a plain list of products and Shopify's `products.json` format (variants, price ranges, tags, options, image alt text).
- **PDF** — read page by page. If a page has real selectable text, it's read as text. If a page is basically a picture (a scanned brochure or a lookbook), the app **renders that page as an image and looks at it** with a vision model instead. You don't have to know or care which kind of PDF you have.

**It copes with real-world messiness:**

- Your price column doesn't have to be called "price". It also recognises *Variant Price*, *Unit Price*, *Sale Price*, *Retail Price*, *Cost*, and *MSRP*. (This matters more than it sounds — if price goes missing, every "under S$200" question breaks.)
- HTML in descriptions is stripped out automatically.
- Any column it doesn't recognise is kept as a spec rather than thrown away.
- Every product gets a stable ID, taken from your SKU or handle where one exists.

**You review before anything else happens.** Extracted products appear in an editable table — fix a typo, add a missing price, delete a bad row, or add a product by hand. Blank fields are treated as *gaps to report*, not as errors.

---

### Behind the scenes — building the "knowledge layer"

When you hit **Confirm & run**, several things happen automatically:

**Grouping.** Products are grouped into sensible categories by what they *are* and *who they're for*, not by keyword matching. Large catalogs are processed in batches of 20, and a follow-up pass merges duplicate labels so you never end up with both "Running Shoe" and "Running Shoes". Anything left over lands in an "Uncategorized" group rather than silently disappearing. **Nothing here is category-specific** — it works out the categories from whatever you upload.

**Deciding what matters.** For each category, the app works out the 8–12 facts a buyer genuinely needs in order to choose — weight, heel drop, surface, support type for shoes; pH, comedogenic rating, format, fragrance for skincare. It derives these from *your* products.

**Working out what applies to what.** Not every fact makes sense for every product. This step marks facts that don't apply so they're never counted as "missing". (Ski poles aren't missing a helmet certification.)

**Cleaning up the facts.** Real catalogs bury everything in one blob:

> `Weight: 258g (US M9) | Heel-to-toe drop: 8mm | Surface: road | Upper: mesh (high ventilation)`

Splitting on `|` works on that file and breaks on the next brand's. So the app uses the AI to map each product onto the fixed list of facts instead — turning `mesh (high ventilation)` into `ventilation: high`, something no delimiter trick could do.

**Then it checks its own work.** Every single extracted value must appear *word-for-word* in that product's own source text. If it doesn't, the value is thrown away and counted as "dropped". **A made-up spec cannot get into the system, even if the AI produces one.** Your original spec text is never overwritten — it's kept alongside.

If any of this fails, the app falls back a level rather than breaking: no schema means it uses your raw specs; a failed batch is retried once and then left alone; and everything the run dropped or skipped is reported in the progress log and on the dashboard.

---

### Tab 2 — Dashboard

**One headline number: your Catalogue Readiness Score.** How likely an AI assistant is to be able to work with your content. It's calculated the same way every time — no AI guesswork in the score itself:

```
Readiness = 25% attribute_completeness
          + 20% persona_coverage
          + 15% not_for_coverage
          + 15% comparative_context
          + 25% claim_grounding
```

| Ingredient | Weight | Plain meaning |
|---|---|---|
| Attribute completeness | 25% | How many of the facts that matter you actually supply |
| Persona coverage | 20% | Whether the content speaks to a range of different shoppers |
| "Not for" coverage | 15% | Whether it says who the product *isn't* right for |
| Comparative context | 15% | Whether it explains how the product compares to its siblings |
| Claim grounding | 25% | How much of the written content is backed by real evidence |

Two deliberate design choices worth knowing:

- **A product that suits everybody ranks for nobody.** Content with no "poor fit" audience and no exclusions gets marked down, because an assistant needs a reason to *not* pick something.
- **"Fit" is reported separately and never blended in.** If only 2 of your 20 shoes suit marathon runners, that's a *merchandising* outcome, not a content gap — no amount of rewriting makes a 340g shoe lightweight. Mixing the two would produce a number that means neither.

**"What your catalogue can't answer".** Since the app refuses to invent facts, the only honest fix for genuinely missing data is to go and get it. So you get a work order:

- **By attribute** — ranked by how many products are missing it, with a reason explaining which shopper need depends on it.
- **By product** — for anyone fixing one item at a time.

Both download as CSV, ready to hand to whoever owns the product data.

**Per-cluster breakdown.** Completeness, readiness, persona fit, which facts are missing, how many products were successfully cleaned up, how many values got dropped as unverifiable, and a per-product drill-down.

---

### Tab 3 — Personas

For each category the app suggests **2–3 realistic shopper types**, written the way people actually talk to AI assistants — not generic marketing personas.

Each one is then turned into **machine-checkable rules** (e.g. `weight_grams lt 250`, `surface contains road`) and every product is tested against them **in pure Python** — deterministic, repeatable, inspectable, and free of subjective LLM ratings. Open any persona to see exactly which products passed, which failed, and which couldn't be judged.

This produces two separate numbers, and keeping them apart is the whole point:

- **Coverage** — *can we even answer this shopper's questions?* A content problem. You fix it by supplying data.
- **Fit** — *given what we know, does the product actually suit them?* A merchandising problem. You fix it by stocking different products.

Pick one persona per category and the app writes a short user story (*"As a …, I need … so that …"*), then generates content for every product in that group.

---

### Tab 4 — Generated Content

For each product you get a full, copy-and-paste-ready bundle:

- **A one-line pitch, a "best for", and an FAQ**
- **Audiences** — labelled strong / partial / **poor** fit (at least one poor fit is mandatory)
- **Derived insights** — 3–5 useful conclusions like "suited to warm-weather running", each showing its reasoning and citing the exact fact it came from
- **Shopper questions** — 3–4 questions this product can now answer, with answers
- **"Not for"** — at least two clear exclusions, each with evidence
- **Use cases** — real scenarios, each pointing at the spec that justifies it
- **Comparisons** — against the closest-priced *real* products in your own catalogue, never invented rivals
- **A flagged list** — anything that couldn't be proven, quarantined for human review rather than published

**How it refuses to make things up.** This is the part most worth understanding. Generation may only use the verified facts, and everything it writes then passes three separate checks:

1. **Evidence check** — every claim must cite an exact fact name *and* its exact value. Cite something that doesn't match your data and the claim is deleted.
2. **Reasoning check** — a second, stricter AI pass asks: *does this evidence actually prove this claim?* A price and a weight do not prove humidity suitability. A "daily trainer" label does not prove marathon suitability. When in doubt, it rejects.
3. **Comparison check** — done in plain code. The rival product must be real, the fact must exist on *both* products, the quoted numbers must match your data exactly, and the "lighter / heavier / similar" direction is recalculated rather than trusted.

Anything that fails goes into the flagged list with a reason, shown on screen as *"Flagged for human review before publishing"*. **And if a product has no verified facts at all, the app writes nothing and says so** — rather than producing confident-sounding fiction.

**Exporting — the part a brand actually plugs in.** Three formats, three audiences:

- **CSV** — for merchandisers, or re-importing into a catalog / PIM system.
- **JSON** — for developers, carrying the complete structured knowledge layer.
- **schema.org JSON-LD** — *the one that matters.* Standard `Product` markup that drops straight into a product page as a `<script>` tag. Your verified facts become `additionalProperty` entries, which is exactly the format AI shopping surfaces and crawlers already read. The content stops living in this app and starts living on your site.

Currency is auto-detected from your price strings (S$, US$, £, €, ¥, RM, ₹, A$), and stock availability is only ever declared when your catalogue actually proves it.

---

### The "Ask like a Shopper" tab — the proof

This is where the app tests its own work instead of asking you to take its word for it.

Type a real question and here's what happens:

**1. The question is taken apart.** Into the *task* (what they're doing), the *context* (humid weather, oily skin), *hard requirements* (under S$200), *preferences* (lightweight), and *derived needs* (humid weather implies a need for ventilation). Every piece quotes the exact words it came from, so nothing is silently invented.

Crucially, the app then **removes invented precision**:

- If it decided you want "stability" shoes but you never said the word "stability", that's demoted from a requirement to a preference.
- If it invented a number you never gave — "low drop" becoming "under 6mm" — the number is stripped out and it's ranked as a relative preference instead.
- Vague words with no threshold ("lightweight", "excellent support") are never treated as pass/fail tests.

Every downgrade is reported back to you as an *ambiguity*, so you can see how your question was read.

**2. Both versions of your catalogue are searched (dual indexing).** Products are converted into number-vectors and compared by meaning, not keywords. Both sides cover **exactly the same products**, so the only thing that differs is the content itself — searching your whole catalogue on one side and a subset on the other would rig the result, and the app says so on screen if you haven't generated content for everything yet.

*(No vector database, deliberately. At catalogue scale — hundreds to a few thousand products — a plain NumPy dot product takes under a millisecond, adds no dependency, and keeps deployment small and cold starts fast.)*

**3. Hard requirements are enforced in code, not by vibes.** A budget ceiling of S$200 rules out a S$219 shoe, full stop — the AI is not allowed to talk its way past it. The check is also smart about naming: if one category calls it `support` and another calls it `support_type`, both are matched.

**4. Common-sense guardrails.** Ask about humid weather and a product with no ventilation, breathability, or moisture-wicking evidence won't be recommended — it will say *"the available content does not establish suitability for the requested hot or humid conditions"*. Ask about a half marathon and a gym or indoor shoe won't sneak in just because it's cheap and light.

**5. Results are ranked with reasons.** For each product you see whether it's recommended, one sentence explaining why, which specific facts were cited, which requirements it fails, and — most usefully — **what your content still can't answer**.

**6. Before and after, side by side.** Plus a callout of what moved: *"Vela Aero 9: not surfaced at all → #2"*.

The safety net matters here too: the AI ranker is never allowed to overturn a requirement that code already verified, and if its response comes back malformed the app shows the plain search order and clearly labels it as degraded — rather than passing off a broken ranking as a real one.

---

## Key features at a glance

- **Multi-format extraction** — CSV, XLSX, JSON, PDF, and Shopify exports, parsed and normalised automatically
- **Two-pass schema inference** — one pass fixes the canonical attribute names per category, a second maps every product onto them, so key names can't drift and completeness scoring stays meaningful
- **Verbatim verification** — every extracted value must exist word-for-word in the source text; no hallucinated specs
- **Deterministic fit scoring** — personas become machine-checkable criteria evaluated in pure Python, not subjective LLM ratings
- **Objective readiness score** — five fixed, weighted components; identical inputs always give an identical score
- **Dual indexing** — raw catalogue vs. agent-optimised content, for a genuine before/after comparison
- **Constraint-aware ranking** — hard constraints gated in code before the model ever ranks
- **Gap reporting** — downloadable CSV work orders for the data that's genuinely missing
- **Export formats** — CSV (merchandisers), JSON (developers), schema.org JSON-LD (embed in product pages)

---

## Sample data

Two demo catalogues ship with the app, chosen to prove the system generalises across very different categories:

- **Running shoes (20 products)** — full technical specs (weight, drop, stack height, surface, support, fit, upper) on some products, and a bare handful of fields on others.
- **Facial skincare (20 products)** — pH, comedogenic rating, format, fragrance and ingredient lists, again unevenly supplied.

Both carry an **intentional data-quality gradient**: richly-specified products sit alongside near-empty ones, so the readiness score and gap report have something real to find. The demos are not cherry-picked to look good.

---

## Technology stack

- **Language:** Python 3.10+
- **UI:** Streamlit (deployed on Streamlit Community Cloud)
- **LLM & reasoning:** OpenAI API — `gpt-4o-mini` for text *and* vision
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Similarity search:** NumPy (L2-normalised matrix, cosine similarity by dot product — no vector DB)
- **Data processing:** pandas, openpyxl (XLSX)
- **PDF handling:** pdfplumber (text extraction) + PyMuPDF/`fitz` (page-to-image rendering for the vision fallback)
- **Validation:** custom deterministic Python checks for fit scoring, constraint gating, and comparison verification
- **Caching:** SHA256-keyed disk cache of LLM results, plus a spend ledger and hard budget cap

Both model choices live in `app/llm_utils.py` if you want to swap them.

---

## Cost, safety, and privacy

- **Every AI call is cached to disk.** Re-run the same catalogue and it costs nothing after the first pass.
- **A hard spending cap** (default **$5**, set `AGENT_BUDGET_USD` to change it) stops a runaway loop burning through credit. Delete `.cache/spend.json` to reset the counter.
- **A live sidebar** shows calls made, cache hits, estimated cost, and tokens used *this session*.
- **Your API key is never displayed** anywhere in the interface.
- **The app discloses in-app** that uploaded catalogue content is sent to OpenAI for processing.
- **No login and no database.** Everything lives in the browser session and disappears when you close it.
- **A live progress log** on every long-running step shows exactly which stage is running, so you're never staring at an unexplained percentage.

---

## Project layout

```
app/
  app.py            The Streamlit interface and all five tabs
  extraction.py     Reads CSV / Excel / JSON / PDF catalogues into products
  normalization.py  Turns messy spec text into clean, verified facts
  pipeline.py       Grouping, gap analysis, personas, content generation, scoring
  fit.py            Turns a persona into rules and tests every product against them
  gaps.py           Builds the "go and measure this" data-request lists
  retrieval.py      Search, question understanding, and constraint-aware ranking
  export.py         CSV, JSON, and schema.org JSON-LD output
  ui.py             Progress bar with a live log
  requirements.txt
data/
  shopify_running_shoes.csv      20-product demo catalogue
  shopify_facial_skincare.csv    20-product demo catalogue
README_instructions.md           The challenge brief, in plain language
```

---

## Performance notes

- **Latency:** roughly 1–3 minutes for a 20-product catalogue on a first run — grouping, cleaning, personas and per-product writing each cost a model round trip. The progress log tells you where you are.
- **Second runs are near-instant** thanks to the disk cache.
- **Large catalogues:** processing is serial, so 200+ products will be noticeably slower.
- **Needs an internet connection and an API key.** Every stage calls a model; there is no offline mode.

---

## Known limitations

- **Serial processing.** Extraction, clustering, schema inference, personas and content generation all run one after another. Batching them concurrently is the obvious next win.
- **No vector database.** Embeddings live in NumPy arrays — ideal at catalogue scale, but 10k+ products would need a real index.
- **Session-based.** No persistent storage; reloading the page clears the analysis.
- **Duplicate cluster names can lose data.** Clusters are keyed by name, so if multi-batch merging produces two groups with the same label, the second overwrites the first. Planned fix: switch to embeddings + agglomerative clustering, or key clusters by ID.
- **Garbage in, honest out.** If your catalogue has almost no specs, the app won't invent any. You'll get a low score and a precise list of what to go and measure — which is the useful answer, not a failure.

---

## Third-party acknowledgments

**Libraries & frameworks**

- **Streamlit** — interactive web UI framework
- **pandas** — data manipulation and analysis
- **NumPy** — numerical computation and similarity search
- **pdfplumber** — PDF text extraction
- **PyMuPDF (`fitz`)** — PDF page rendering for the vision fallback
- **openpyxl** — XLSX parsing
- **Pillow** — image handling

**APIs & services**

- **OpenAI API** — `gpt-4o-mini` (text generation, vision understanding) and `text-embedding-3-small` (embeddings), used for schema inference, clustering, persona generation, content generation, intent parsing and ranking. Cost managed via SHA256-keyed disk caching and a hard budget guard.
- **Streamlit Community Cloud** — application hosting and deployment

**Datasets**

- **Shopify product CSV examples** — running shoes and facial skincare sample catalogues (20 products each), authored with an intentional data-quality gradient to demonstrate category-agnostic clustering and gap analysis.

---

## License

Built for LifeHack 2026 — Rezolve AI Challenge.

## Questions?

Open an issue on GitHub, or see `README_instructions.md` for the original challenge brief.
