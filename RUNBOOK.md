# Runbook

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                              # then paste your key
```

## Run it

```bash
set PYTHONPATH=src

python -m agentcard doctor                # what can this machine actually reach?
python -m agentcard all                   # enrich -> index -> simulate
streamlit run app/streamlit_app.py
```

`doctor` probes the OpenAI API, prints spend so far, and — this is the part
worth running before anything else — reports whether your key can actually see
`gpt-4.1`, `gpt-4.1-mini` and `text-embedding-3-small`, suggesting alternatives
for any it cannot. Model availability varies by account and by project key.
Finding out here costs one list call; finding out during enrichment costs a
failed run.

Every command falls back to the local provider automatically if the API is
unreachable, and says so on stderr rather than failing.

To force one or the other:

```bash
python -m agentcard all --provider local --embed-provider local   # no network, free
python -m agentcard all --provider openai --embed-provider openai # the real thing
```

Ask a single question without the UI:

```bash
python -m agentcard ask "I have wide feet and need a daily trainer under S$200"
```

## What runs on what

| Command | Cost | Needs network |
|---|---|---|
| `doctor` | free | no |
| `all --provider local` | free | no |
| `all --provider openai` | ~$0.30–0.80 for 40 products | yes |
| `streamlit run …` | free unless embeddings are set to openai | no |

The demo reads pre-generated JSON from `out/`. Nothing is enriched at the booth.

## Order of work with 20 hours left

1. **Get real enrichment producing valid cards.** Everything is downstream of
   this. `python -m agentcard enrich --provider openai --limit 2` first — two
   products, then look at the JSON before spending on forty.
2. Re-index and re-run the simulator on real embeddings. The local numbers are
   a lower bound; `text-embedding-3-small` should hold or improve the lift and
   will make the `enriched` arm honest rather than lexical.
3. Rehearse the live demo with queries nobody has typed before. The parser is
   config-driven — if a judge's phrasing misses, the fix is a line in
   `configs/*.yaml`, not code.
4. Only then touch the UI.

## Running both catalogue variants

```bash
python -m agentcard compare
```

Runs the whole pipeline against `data/raw/spec_rich` and `data/raw/typical`,
then prints both lifts and writes `out/variant_comparison.json`, which the
Evidence tab renders. The point is not to pick the flattering number: a brand
already publishing structured specs gets less from enrichment than one
publishing marketing copy and a price, and the second is the common case.

## Pointing it at a different model

```bash
AGENTCARD_LLM_BASE_URL=https://your-endpoint/v1
AGENTCARD_EMBED_BASE_URL=https://your-endpoint/v1
```

Any OpenAI-compatible endpoint. The enricher needs chat completions with
structured outputs; the retriever needs embeddings. `doctor` prints which
endpoints are in use.

## Booth script

1. Show a raw catalogue row. "This is what an agent sees today. Readiness 24."
2. Show the Agent Card beside it. "Same product. Readiness 82. Here's what got
   added, and here's where every field came from."
3. Take a query from a judge. Show the parsed constraints before the results —
   the reasoning trace is more convincing than the ranking.
4. Show the two result columns. "Same embedder, same ranker. The only
   difference is what the product says about itself."
5. Ask it something the catalogue does not sell — "my hands hurt from the gym".
   It declines, and explains that the best match fell below a floor calibrated
   from the query set. Knowing when *not* to recommend is the part most demos
   skip.
6. Open `configs/skincare_facial.yaml`. "Adding a category is this file."

## Failure modes and what to say

**Structured outputs rejects the schema.** `schema.to_strict()` handles the
draft-07 → strict-mode conversion (every object gets
`additionalProperties: false` and a complete `required` list; unsupported
keywords are stripped; `category_extension` becomes a JSON string that
`enrich.postprocess()` parses back). If a model version tightens further, that
one function is where to fix it.

**A card fails schema validation.** The runner logs it and keeps going rather
than dying at product 23 of 40. `out/enrichment_summary.json` lists every
failure with its reason.

**Conference wifi dies.** `--provider local --embed-provider local` runs the
entire demo offline, and the UI reads from `out/` regardless. Say it out loud:
the demo does not call an API to answer a query, because a shopping agent
shouldn't have to.

**A judge's query returns something wrong.** Show them the parsed filters. Being
able to say *why* it was wrong — "it read 'pronation control' as stability, and
the fix is one line of YAML" — reads better than a system that is never wrong.

## Vectors on disk

`out/vectors_page.npy` and `out/vectors_card.npy` — two per product. If you are
carrying an older run, delete both plus `out/catalogue.db` before re-indexing;
the previous layout used different filenames and a stale pair will not be
picked up.

## If the numbers move after re-indexing

`python -m agentcard index` and `python -m agentcard simulate` must be re-run
together. The decline floor is written by `simulate` and read by the retriever,
and it is only valid for the embedding provider that produced it — the app
ignores a calibration from a different provider rather than declining on a
threshold that means nothing.

## Tests

```bash
python -m pytest tests -q
```

Twenty checks, including: every card validates against the schema, every card
carries at least two `not_for` entries, enrichment never scores worse than the
raw baseline, out-of-stock products are never recommended, budget ceilings are
respected, and the hard filter never *lowers* recall. That last one caught a
real regression.

`tests/test_openai_path.py` stubs the API client and asserts the request shape,
that an identical second call is served from cache, that the budget ceiling
actually raises, and that a strict-mode response round-trips into a card that
validates. It runs offline, so the first live run fails for interesting reasons
rather than plumbing ones.
