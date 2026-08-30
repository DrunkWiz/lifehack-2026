## 💡 Inspiration

AI agents fail **silently** on incomplete catalogs. Retailers don't know what's broken or where to start fixing it.

We wanted to measure catalog readiness the way you'd measure any product: **objectively, with a score and a fix list.**

---

## ⚙️ What it does

**AgentShopper** audits product catalogs and scores how well they support AI agents.

It extracts data from any format (CSV, JSON, PDF), infers optimal schemas, grades fit against real personas using deterministic Python checks, and exports gap reports showing _exactly_ what to fix.

---

## 🔧 How we built it

- **📥 Extraction & normalization** — Multi-format ingestion with two-pass schema inference and verbatim verification
- **🎯 Deterministic fit scoring** — Convert personas into machine-checkable criteria, then run Python evaluation (not LLM guesses)
- **🔀 Dual indexing** — Before/after catalog comparison showing impact of fixes
- **🖥️ Streamlit UI** — Interactive analysis across 5 tabs (Extract, Ask, Dashboard, Personas, Content)
- **📤 Export formats** — CSV for merchandisers, JSON for developers, JSON-LD for embedding

---

## 🧗 Challenges we ran into

- Schema inference from **sparse, inconsistent** product data without overfitting
- Preventing LLM hallucination while still capturing useful reasoning
- Managing LLM costs at scale with smart caching and budget guards
- Complex UI state management across a **7-step** analysis pipeline
- Avoiding silent data loss from duplicate cluster names

---

## 🏆 Accomplishments that we're proud of

- ✅ Built a system that gives you a readiness score **_and_** tells you exactly why
- ✅ Deterministic fit scoring removes guesswork from catalog evaluation
- ✅ Dual indexing proves the concrete impact of fixing data gaps
- ✅ Category-agnostic design means it works on any catalog, **day one**
- ✅ Caught and documented real architectural bugs in production code

---

## 🧠 What we learned

> **LLM validation layers matter as much as the models themselves.**

Deterministic grounding beats magic. Before/after comparison is how you make insights actionable.

And schema inference requires **two passes** — first to find what's _possible_, second to verify what's _real_.

---

## 🚀 What's next

- **Scale** to larger catalogs
- **Predict** which gaps matter most with predictive models
- **Integrate** live agent feedback loops
- **Expand** export formats for different stakeholder needs
