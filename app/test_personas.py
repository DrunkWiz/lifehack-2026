"""
Standalone test for the pipeline — no Streamlit UI needed.

Run:
    python test_personas.py

Requires .streamlit/secrets.toml with OPENAI_API_KEY (same as the app uses),
since llm_utils reads from st.secrets.

What this does:
1. Defines a small hardcoded product cluster (running shoes) with realistic
   specs — some attributes present, some deliberately missing.
2. Runs: expected attributes -> attribute completeness -> persona candidates
   -> user story -> rich agent content (personas incl. poor-fit, not_for,
   comparisons, provenance) -> rendered passage -> readiness score before/after.
3. Prints everything so you can eyeball whether the output makes sense.
"""

from pipeline import (
    determine_expected_attributes,
    attribute_completeness,
    suggest_personas,
    generate_user_story,
    competitors_for,
    generate_agent_content,
    render_passage,
    score_components,
    readiness_score,
    top_gaps,
    ask_confidence,
)

CLUSTER_NAME = "Running Shoes"

SAMPLE_PRODUCTS = [
    {
        "name": "AeroFlex Trail Runner",
        "price": "S$179",
        "description": "Lightweight trail running shoe with engineered mesh upper for breathability.",
        "specs": {
            "weight": "210g",
            "breathability": "85% engineered mesh",
            "drop": "8mm",
            "cushioning": "medium",
        },
    },
    {
        "name": "Basic Jogger X",
        "price": "S$99",
        "description": "Everyday running shoe.",
        "specs": {
            "color": "black",
        },
    },
]


def main():
    print(f"\n=== Cluster: {CLUSTER_NAME} ===")
    print(f"Products: {[p['name'] for p in SAMPLE_PRODUCTS]}\n")

    print("--- Step 1: Expected attributes ---")
    expected_attrs = determine_expected_attributes(CLUSTER_NAME, SAMPLE_PRODUCTS)
    print(expected_attrs, "\n")

    print("--- Step 2: Attribute completeness ---")
    avg_completeness, per_product, missing_counts = attribute_completeness(SAMPLE_PRODUCTS, expected_attrs)
    print(f"Cluster average: {avg_completeness}%")
    for p in per_product:
        print(f"  - {p['name']}: {p['completeness_pct']}% | missing: {p['missing']}")
    print()

    print("--- Step 3: Persona candidates ---")
    personas = suggest_personas(CLUSTER_NAME, expected_attrs, per_product)
    for i, persona in enumerate(personas):
        print(f"  [{i}] {persona['title']} — rating {persona.get('persona_rating_pct')}%")
    print()

    if not personas:
        print("No personas returned — stopping here.")
        return

    chosen = max(personas, key=lambda p: p.get("persona_rating_pct", 0))
    print(f"--- Step 4: Chosen persona: '{chosen['title']}' ---\n")

    story = generate_user_story(CLUSTER_NAME, chosen)
    print(f"User story: {story}\n")

    print("--- Step 5: Rich agent content (first product) ---")
    product = SAMPLE_PRODUCTS[0]
    comps = competitors_for(product, SAMPLE_PRODUCTS)
    agent_content = generate_agent_content(product, CLUSTER_NAME, chosen, story, comps)
    print("Personas:", agent_content.get("personas"))
    print("Not for:", agent_content.get("not_for"))
    print("Comparisons:", agent_content.get("comparisons"))
    print("Unsupported claims:", agent_content.get("unsupported_claims"))
    print()

    print("--- Step 6: Rendered passage ---")
    print(render_passage(agent_content))
    print()

    print("--- Step 7: Readiness score ---")
    completeness_pct = per_product[0]["completeness_pct"]
    before = readiness_score(completeness_pct, None)
    after = readiness_score(completeness_pct, agent_content)
    print(f"Before: {before}%  ->  After: {after}%")
    print("Component breakdown (after):", score_components(completeness_pct, agent_content))
    print("Top gaps (after):", top_gaps(completeness_pct, agent_content, per_product[0]["missing"]))
    print()

    print("--- Step 8: Ask a query against this product's content ---")
    passage_text = render_passage(agent_content)
    query = "half marathon shoes for humid weather under S$200"
    result = ask_confidence(query, product["name"], passage_text)
    print(f"Query: {query}")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
