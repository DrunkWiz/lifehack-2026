# Architecture

## The gap, stated precisely

An AI shopping assistant fails on a catalogue for three separate reasons, and
they need three different fixes:

| Failure | What's missing | Fixed by |
|---|---|---|
| It can't tell if the product fits | Typed, filterable attributes with units | `hard_constraints` |
| It can't connect the shopper's words to the spec | A bridge from lifestyle language to attributes | `situational_tags`, `use_cases` |
| It won't commit to a recommendation | Grounds to *rule products out* | `not_for`, `comparisons`, `provenance` |

Marketing copy addresses none of the three, because it was written to persuade a
human who has already landed on the page. The Agent Card is a parallel
representation written for the machine that decides whether the human ever
sees the page at all.

## Pipeline

```mermaid
flowchart LR
  A[Shopify CSV<br/>title, body, tags, spec blob] --> B[ingest]
  B --> C[enrich<br/>structured outputs against agent_card_schema.json]
  CFG[configs/*.yaml<br/>category definition] --> C
  C --> D[Agent Cards]
  B --> R[raw baseline card]
  D --> E[readiness<br/>deterministic formula]
  R --> E
  D --> F[index<br/>SQLite constraints + embeddings]
  A --> F2[raw corpus embeddings]
  F --> G[hybrid retrieval<br/>SQL filter → embedding rank]
  F2 --> S[simulator]
  G --> S[simulator<br/>recall@3, three arms]
  G --> UI[Streamlit demo]
```

## The four decisions worth defending

**1. Hard filter before semantic rank, not after.**
Embeddings cannot represent "under S$200" or "not for wide feet" — they can
only sit near those phrases. So typed constraints run first as SQL and
eliminate products that are genuinely wrong; the embedding only ever ranks
survivors. Reversing the order means a beautifully similar product that costs
S$329 outranks a correct one.

**2. Hard filters only where a mismatch is disqualifying.**
`hard_filter_keys` in each category config draws the line. Wrong surface or
wrong width makes a shoe wrong. A neutral shoe for a mild overpronator is
merely less preferred — so `arch_support` and `skin_type` boost the ranking
rather than eliminating candidates. Early on we filtered on everything and
recall went *down*: the parser inferred `arch_support = stability` from
"pronation control" and eliminated the motion-control shoe that was the correct
answer. The split is in the config, so it is a tuning decision, not a rewrite.

**3. Negative information is a required field, not a nice-to-have.**
`not_for` is the one field no marketer writes and the one an agent needs most.
The readiness score penalises cards whose personas are all positive, because a
card that suits everyone helps rank nothing. Two `not_for` entries is a
schema-level expectation enforced in the tests.

**4. Readiness is a formula, not a model.**
Five weighted components computed in Python over the card. A judge's first
question about any score is whether it's just another model guessing; the
answer is `readiness.py`, forty lines, deterministic, same input same output.

## Generalisability

The core schema never changes. A category is a ~20-line YAML file declaring
required attributes, situational vocabulary, persona axes, common exclusions,
the synonym map from shopper language to constraint values, and which
constraints are allowed to eliminate a product.

Nothing in `retrieve.py`, `readiness.py`, `index.py` or the UI knows what a
running shoe is. `configs/footwear_running.yaml` and
`configs/skincare_facial.yaml` are the entire difference between the two
categories in this demo, and the simulator reports both separately so the
claim is checkable rather than asserted.

## Measurement

Ground truth by construction: every test query is generated *from* a known
product without naming it, so the right answer is known and retrieval has to
find its way back. Three arms share an embedder, a ranker and a query set:

| Arm | Corpus | What it isolates |
|---|---|---|
| `raw` | the product page vector | today's catalogue |
| `enriched` | page **and** card vectors, best of the two | what the added content buys |
| `enriched+sql` | the same, constraints filtered first | what the *structure* buys on top |

**How the page and the card are joined took two attempts, and the second one is
the interesting part of this section.**

Indexing the card *instead of* the page threw away the brand's own copy, so
enrichment could only lose information. Concatenating them failed in a subtler
way: an embedding is normalised over the whole document, so bolting 2,600
characters of card onto 500 characters of page dilutes a page that already
matched the query. On skincare — where the raw rows already publish pH and
comedogenic ratings, so the page often matched — the concatenated arm scored
*below* raw. Adding correct information made retrieval worse, which should not
be possible and was a signal the join was wrong rather than the content.

Each product now carries two vectors, page and card, and scores as the better
of the two. Standard multi-field retrieval, and it states the true relationship:
the card is an additional document about the product, not a replacement for the
page and not an extension of it.

This does not rig the result. Every product gains a card, so a rival's card can
outrank yours — only the per-product score is monotone, never the ranking. Raw
recall can and does move between runs.

Reported as recall@3 and MRR, split by category. The `raw → enriched` gap is
the content argument; the `enriched → enriched+sql` gap is the architecture
argument. Keeping them separate is what makes the number believable.

## Knowing when to decline

An assistant that answers everything is one you cannot trust on anything. The
simulator records the similarity at which retrieval actually found the correct
product across the whole query set and writes the 5th percentile to
`out/retrieval_calibration.json`. A query whose best match falls below that is
doing worse than the worst genuine match ever observed, and the UI says so
rather than returning the three least-bad products.

The floor is measured, not chosen, and it is tagged with the embedding provider
that produced it — absolute cosine thresholds do not transfer between models, so
a calibration from a different provider is ignored rather than trusted.

## Brand adoptability

The input is an unmodified Shopify product export. The output is JSON against a
published schema plus a SQLite table any existing stack can query. There is no
proprietary index and no vector database to adopt: a brand can run the
enrichment, keep the cards in their own CMS, and serve them from a feed.

Three integration shapes, in ascending order of commitment:

1. **Upload** a CSV, get cards and readiness scores back — one afternoon.
2. **Scheduled feed** — re-enrich on catalogue change, cached by content hash
   so only changed products cost anything.
3. **Live endpoint** — serve the Agent Card at a well-known URL per product so
   agent crawlers read the structured version instead of scraping the page.

## Cost control

Every completion and every embedding is cached to disk by SHA-256 of the exact
input. Re-running after a crash costs nothing. A spend ledger tracks per-model
usage and the runner refuses to start a call that would cross
`AGENTCARD_BUDGET_USD`. Enriching 40 products with `gpt-4.1` plus query
generation on a mini model and `text-embedding-3-small` lands well under a
dollar; the cap exists for the runs you didn't mean to start.
