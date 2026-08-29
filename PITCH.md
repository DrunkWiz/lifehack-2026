# Pitch notes

Speaking aid, not a script. Numbers are from the `spec_rich` run; re-check them
against `out/simulator_report.json` and `out/variant_comparison.json` before you
go up.

## Open (30 seconds)

Adobe's US retail dataset — over a trillion site visits — has AI referrals up
**393% year on year in Q1/26**, and **39% of online shoppers** now use an AI
assistant somewhere in their journey. Shopee went live inside ChatGPT this
month.

And AI-referred shoppers are not just more numerous, they are worth more:
**+42% conversion rate**, **+48% time on site**, **+13% pages per visit**
against non-AI traffic. *(All figures from Rezolve's own brief, citing Adobe
Analytics and Adobe Digital Insights.)*

So the shelf is moving, and the traffic arriving from it converts better than
the traffic brands already optimise for. The question is what a brand's product
content has to look like when the shopper never sees the page — when an agent
reads it and decides, on their behalf, whether to mention the product at all.

Frame the cost that way at the booth: enrichment ran at **$0.018 a product**.
Against a 42% conversion premium on the channel growing fastest, the question
is not whether a brand can afford this — it is why their catalogue is still
written for a shopper who is no longer the one reading it.

## The gap (60 seconds)

An assistant fails on a catalogue for three separate reasons:

1. **It can't tell if the product fits.** No typed, filterable attributes with
   units — "lightweight" is not a weight.
2. **It can't connect the shopper's words to the spec.** The shopper says
   "humid Singapore weather"; the catalogue says "engineered mesh". Nothing
   bridges them.
3. **It won't commit.** Nothing in the content lets it rule a product *out*, and
   an assistant that can't exclude can't confidently recommend.

Marketing copy addresses none of the three, because it was written to persuade
a human who already landed on the page.

## What we built

An **Answer Engine Optimisation layer for catalogues**. Three parts:

- **Agent Cards** — a published JSON schema that carries typed constraints,
  situational tags, use cases grounded in specs, personas including poor fits,
  comparisons with mandatory tradeoffs, and provenance on every field.
- **A readiness score** — 0–100, five weighted components, computed
  deterministically in Python. *Not* a model grading a model.
- **A simulator** — generates intent queries from known products without naming
  them, then measures whether retrieval finds its way back.

## Demo (4 minutes)

1. **Before/After tab.** A raw catalogue row. "This is what an agent sees today.
   Readiness 24." Then the card beside it at 71+, with the provenance table.
2. **Ask tab.** Take a query from the judge. Show the parsed constraints *before*
   the results — the reasoning trace convinces more than the ranking does.
3. Toggle the comparison column. "Same embedder, same ranker. The only
   difference is what the product says about itself."
4. **Compare tab.** Two shoes side by side, the tradeoff the enricher wrote, and
   who each one is wrong for. Then type a shopper into the box and let it pick.
5. Ask it something the catalogue doesn't sell. It declines, against a floor
   calibrated from the query set. *Knowing when not to recommend is the part
   most demos skip.*
6. Open `configs/skincare_facial.yaml`. "Adding a category is this file."

## Rubric answers

**1. Problem comprehension.** The three failure modes above, each mapped to a
schema field. We can also show where enrichment *doesn't* help: on a catalogue
that already publishes structured specs, the lift is small, because the brand
has already done part of this work by accident.

**2. Architecture.** Hard filter then semantic rank, and we can say why:
embeddings cannot represent "under S$200" — they can only sit near it. Only
constraints that make a product *wrong* eliminate; preferences rank. We learned
that the hard way — filtering on inferred preferences made recall go *down*, and
the fix is a config key, not a rewrite.

**3. Reasoning quality.** 99 generated intent queries, recall@3:
`63.6% raw → 68.7% cards → 72.7% cards + SQL`, MRR 0.465 → 0.566. The two gaps
are separate arguments: content, then structure.

**4. Scalability.** Two categories, two YAML configs, one codebase, reported
separately. Nothing in the retriever knows what a running shoe is.

**5. Adoptability.** Input is an unmodified Shopify export. Output is JSON
against a published schema plus a SQLite table any stack can query. No
proprietary index, no vector database to adopt. And the pipeline speaks the
OpenAI wire format against a configurable base URL — point
`AGENTCARD_LLM_BASE_URL` at a proprietary commerce LLM and the whole thing runs
on it without a code change.

## Where this sits in Rezolve's stack

Their agentic commerce layer names **AEO & Enrichment** under *Help Me Discover
It*, and **Conversational Selling — side-by-side comparison** under *Help Me
Choose It*. That is exactly what these two surfaces do. Their journey slide also
names *Intent-Based Recommendations* — the query-to-constraints path — and
"products in AI answers with **verified** specs and reviews", which is what the
provenance block and the unsupported-claims flag are for.

## What the rubric implies you must show

Neither the deck nor the Devpost page lists deliverables — confirm with the
organisers. But the rubric itself constrains the demo:

- **Rubric 3 is scored "based on live demo"**, so unseen queries must be typed
  in front of a judge and answered. That is not a video or a slide.
- **Rubrics 1 and 2** are scored on comprehension and justification, so someone
  on the team has to explain the architecture and defend a decision — including
  one that went wrong and got fixed.
- **Rubric 4** needs two categories visible side by side, not asserted.
- **Rubric 5** needs an integration path someone can watch working: upload a
  file, get cards and markup out.

## Cost

40 products enriched for **$0.73** on `gpt-4.1`, embeddings on
`text-embedding-3-small`. Every completion cached by SHA-256 of its input, so a
re-run is free. At this rate a 100,000-SKU catalogue is roughly $1,800 as a
one-off, and near zero on subsequent runs since only changed products re-enrich.

## If a judge asks what doesn't work

Say these before they find them:

- The catalogue is synthetic — 40 products, invented brands. We fabricated
  nothing about real companies deliberately. Ingest takes a real Shopify export.
- Query understanding is a synonym map in YAML, not a model. It's fast, cheap
  and inspectable, and it degrades to pure semantic ranking on phrasings we
  didn't anticipate.
- Cards are generated, not reviewed. `provenance.unsupported_claims` flags what
  a human should check before publishing — the compliance story is that the
  system knows what it guessed.
- The readiness score's weights are a judgement call. The formula is
  deterministic; the five components and their weights are ours to defend.
