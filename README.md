# Rezolve AI Challenge — What We Need to Build

Plain-language breakdown of the problem statement.

## The Problem in One Sentence

People are starting to shop by talking to AI assistants instead of typing keywords into search bars, but brands still write product content for websites and Google — so AI agents can't confidently recommend their products.

## Why This Happens

An old-style shopper searches: *"running shoes size 10"*.

A new-style shopper asks an AI: *"I'm training for a half marathon in Singapore's humid weather and need lightweight shoes under S$200."*

To answer the second one, the AI needs to know things a normal product page never says out loud:

- Is this shoe good in heat and humidity?
- Is it right for a beginner half-marathon runner?
- How does it compare to the alternatives?
- Does it fit the budget, and why is it worth it?

Most catalogs only have a title, a price, and a spec list. That's not enough for a machine to reason with, so the product simply doesn't get recommended.

## What We Have to Build

A tool that helps brands **create, improve, or measure** product content so AI shopping assistants can understand and recommend it.

We pick one angle (or combine a few). The suggested options:

| Idea | What it does |
|---|---|
| **AI Content Copilot** | Takes a brand's raw catalog data and rewrites it into descriptions AI agents can actually use. |
| **Content Readiness Score** | Grades a product and says how likely an AI is to recommend it, plus what's missing. |
| **Persona-Aware Generator** | Writes different versions of the content for different types of shoppers and intents. |
| **Simulation Platform** | Fires thousands of realistic shopper questions at a product to find the gaps in its content. |
| **Structured Knowledge Layer** | Converts marketing material into machine-readable data that AI commerce systems can query. |

Anything else that helps a brand get found and picked by AI agents is also fair game.

## Questions Our Solution Should Answer

- What information does an AI agent actually need before it will recommend a product?
- How should a brand describe a product beyond the title and the spec sheet?
- How do we represent attributes, personas, use cases, comparisons, and storytelling in a form an AI can reason over?
- How does a brand know if its content is "AI-ready"?
- Can generative AI do this transformation automatically, at catalog scale?

## How We'll Be Judged

1. **Problem Comprehension** — Do we clearly understand *why* AI agents struggle with product content, and have we named the real gap we're closing?
2. **Solution Architecture** — Is the system well designed, and can we justify our technical choices?
3. **AI Reasoning Quality (live demo)** — Given real intent-driven questions on the spot, does it surface the right products?
4. **Scalability & Generalisability** — Does it work across product categories and datasets, or only on our hand-picked demo data?
5. **Brand Adoptability** — Could a real brand plug this in without much friction? Is there a clear integration path?

## What This Means for Our Build Plan

- The **live demo must handle unseen, natural-language queries** — judges will test it live, so no hardcoded answers.
- Use **at least two very different product categories** (e.g. footwear and skincare) to prove it generalises.
- Show the **before and after**: raw catalog entry in, agent-ready content out, and evidence the AI now picks it.
- Have a **clear integration story** — an API, a file upload, a feed connector — something a brand could realistically adopt.
- Be able to **explain the "why"**, not just show the UI. Problem comprehension is a scored criterion on its own.
