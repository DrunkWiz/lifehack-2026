"""Deterministic offline stand-in for the LLM.

Why this exists: the enrichment pipeline must be testable, and the demo must be
runnable, without an API key or network egress. The local provider produces a
schema-valid Agent Card from the same inputs using rule-based reasoning over
the catalogue plus a category-knowledge fixture that stands in for what a model
knows about running shoes and skincare.

It is NOT a substitute for the real enricher, and it labels itself as
`local-deterministic-v1` in provenance.model so nobody can mistake one for the
other in the output. Set AGENTCARD_LLM_PROVIDER=openai for the real thing.
"""
from __future__ import annotations
import json, re, datetime
from .config import FIXTURES

_TRUTH = None


def truth() -> dict:
    global _TRUTH
    if _TRUTH is None:
        p = FIXTURES / "ground_truth.json"
        _TRUTH = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _TRUTH


# tag -> how it turns into shopper-facing language
S = {
 # --- running ---
 "daily_training": ("Everyday easy mileage, four to five runs a week",
   "Regular runner logging 30-50km a week", "runners who only race and never train easy",
   "there are lighter, more aggressive options for pure race use"),
 "humid_climate": ("Running in tropical humidity where the upper has to breathe and drain",
   "Runner training year-round in a hot, humid climate", "cold or wet-weather running",
   "the open mesh that vents well in heat offers no protection in cold rain"),
 "tropical_training": ("Sustained training blocks in 30°C and 80%+ humidity",
   "Southeast Asia based runner", "cold-climate winter training", "no insulation or water resistance"),
 "beginner_friendly": ("First structured training block, building up from nothing",
   "Beginner runner in their first 6 months", "experienced runners chasing marginal gains",
   "forgiving rather than efficient; faster runners will want more responsiveness"),
 "beginner_runner": ("New runner finding their footing", "Complete beginner", "beginners",
   "demands a developed foot and calf; new runners risk overloading the achilles"),
 "wide_feet": ("Needing genuine room across the forefoot without sizing up",
   "Runner with a wide (2E+) forefoot", "narrow feet", "the roomy last leaves narrow feet sliding"),
 "narrow_feet": ("A locked-in fit for a low-volume foot", "Runner with narrow, low-volume feet",
   "wide feet", "the last is narrow through the midfoot"),
 "race_day": ("Race morning, chasing a personal best over 10km to marathon",
   "Runner targeting a specific race time", "daily training volume",
   "race foams break down quickly under everyday mileage"),
 "road_long_distance": ("Long runs of 20km and beyond on tarmac",
   "Half and full marathon trainee", "short interval sessions", "not built for repeated hard efforts"),
 "tempo": ("Threshold and tempo sessions where turnover matters",
   "Runner doing structured speed work", "easy recovery days", "too firm to take the sting out of tired legs"),
 "recovery_run": ("Easy shakeout the day after a hard session",
   "Runner prioritising recovery between hard days", "race day and fast sessions",
   "the soft midsole absorbs the energy you want returned when running fast"),
 "high_mileage_week": ("Peak weeks above 60km where durability decides everything",
   "High-mileage runner", "occasional runners", "over-built for someone running twice a week"),
 "heavier_runner": ("Carrying more mass, needing a midsole that does not bottom out",
   "Runner above 85kg", "very light runners", "the firm base needs load to compress properly"),
 "heavy_runner": ("", "", "runners above 85kg", "the thin midsole bottoms out under heavier loads"),
 "overpronation": ("Managing inward ankle roll over longer efforts",
   "Runner with mild to moderate overpronation", "neutral runners who dislike guidance",
   "the medial post is intrusive if you do not need it"),
 "flat_feet": ("Low arches that collapse as fatigue sets in",
   "Runner with flat feet or fallen arches", "high-arched runners", "the arch structure will feel like a ridge underfoot"),
 "severe_overpronation": ("Significant pronation control, often on clinical advice",
   "Runner referred for motion control", "neutral or mild pronators", "the correction is aggressive and unnecessary for most"),
 "walking": ("All-day walking and standing rather than running",
   "Walker or shift worker on their feet", "fast running", "the weight penalises turnover"),
 "trail_use": ("Loose, uneven ground where grip decides the run",
   "Trail and off-road runner", "road and treadmill running", "the lugged outsole is loud and unstable on tarmac"),
 "wet_road_grip": ("Wet, greasy surfaces after rain", "Runner in a monsoon climate", "dry hardpack", "the soft rubber wears faster on dry road"),
 "wet_weather": ("Cold rain and puddles", "Wet-climate runner", "hot humid conditions", "the membrane traps heat and sweat above 25°C"),
 "cold_climate": ("Cold-weather running where warmth beats ventilation", "Winter runner", "tropical heat", "insulated construction is unbearable in humidity"),
 "hiking_crossover": ("Doubling as a light hiking shoe on trips", "Runner who also hikes", "technical mountaineering", "no ankle support or rock plate"),
 "travel_light_packing": ("Packing light for a race trip or work travel", "Travelling runner", "everyday cushioned mileage", "minimal cushioning limits time on feet"),
 "budget_daily": ("Getting a dependable trainer without overspending", "Budget-conscious runner", "performance-focused buyers", "the materials are functional rather than premium"),
 "treadmill_indoor": ("Indoor treadmill sessions in an air-conditioned gym", "Gym-based runner", "outdoor trail running", "the outsole compound is tuned for smooth belts"),
 "gym_crossover": ("Mixed gym sessions with short runs attached", "Gym-goer who runs occasionally", "long-distance running", "the low stack offers little protection past 10km"),
 "short_runs": ("Runs under 10km", "Casual runner", "marathon distance", "underfoot protection fades on long efforts"),
 "max_cushion_seekers": ("", "", "runners who want maximum cushioning", "the firm midsole is a deliberate choice, not a soft ride"),
 "long_distance_comfort": ("", "", "runners chasing long-distance comfort", "the minimal stack is unforgiving past 15km"),
 # --- skincare ---
 "oily_skin_humid_climate": ("Oily skin in a humid climate where heavy creams slide off",
   "Oily-skinned shopper in a tropical city", "dry or dehydrated skin", "the oil-controlling base is too light to relieve dryness"),
 "morning_routine_under_5min": ("A morning routine that has to fit in under five minutes",
   "Time-poor shopper who wants three steps maximum", "shoppers who enjoy a long layered routine", "designed to replace steps, not add one"),
 "layerable_under_spf": ("Sitting under sunscreen and makeup without pilling",
   "Shopper layering multiple products", "shoppers using it as a final occlusive step", "too light to seal a routine on its own"),
 "maskne_prone": ("Breakouts along the jaw and cheeks from masks and humidity",
   "Shopper dealing with congestion and breakouts", "dry, flaking skin", "the active dries the surface further"),
 "night_routine": ("The evening step, after cleansing, when skin repairs",
   "Shopper with a distinct PM routine", "morning use before sun exposure", "the active increases photosensitivity"),
 "fine_lines": ("Early fine lines around the eyes and mouth", "Shopper in their 30s addressing early ageing", "shoppers wanting immediate results", "measurable change takes 8-12 weeks"),
 "texture_refining": ("Rough, uneven texture and enlarged pores", "Shopper focused on skin texture", "compromised or irritated barriers", "exfoliating acids worsen an already damaged barrier"),
 "hyperpigmentation": ("Post-acne marks and uneven tone", "Shopper treating pigmentation", "shoppers with active irritation", "acids sting on broken skin"),
 "blackheads": ("Congestion in the T-zone", "Shopper with clogged pores", "dry or sensitive skin", "salicylic acid is drying at this concentration"),
 "sensitive_barrier_repair": ("A compromised, stinging barrier that needs rebuilding",
   "Shopper with sensitised or reactive skin", "shoppers wanting active-driven results", "deliberately free of actives"),
 "post_procedure_recovery": ("The week after a facial, laser or peel", "Shopper recovering from a procedure", "daily long-term use as a sole moisturiser", "the formula prioritises calm over hydration depth"),
 "redness": ("Persistent redness and visible flushing", "Shopper with rosacea-prone skin", "shoppers seeking exfoliation", "no resurfacing action"),
 "air_conditioned_office": ("Eight hours in dry air conditioning", "Office-based shopper", "outdoor humid conditions", "the richer texture feels heavy in humidity"),
 "very_dry_skin": ("Skin that stays tight and flaky even after moisturising", "Shopper with genuinely dry skin", "oily skin", "the occlusive base will feel greasy on oily skin"),
 "dry_skin": ("", "", "shoppers with dry or dehydrated skin", "the formula controls oil rather than adding it"),
 "travel_cabin_size": ("Under 100ml for cabin baggage", "Frequent traveller", "shoppers wanting value per ml", "the small format costs more per millilitre"),
 "refillable_low_waste": ("Cutting packaging waste with a refill system", "Sustainability-minded shopper", "shoppers who want the simplest possible purchase", "refills require buying the vessel first"),
 "acne_prone": ("", "", "acne-prone skin", "the richer emollients raise the comedogenic risk"),
 "mature_skin": ("", "", "mature skin seeking richer hydration", "the gel base is too light for age-related dryness"),
 "pregnancy": ("", "", "anyone pregnant or breastfeeding", "retinoids and high-strength BHA are not recommended in pregnancy"),
 "active_retinoid_use": ("", "", "anyone already using a prescription retinoid", "stacking exfoliating acids on a retinoid routine causes irritation"),
 "fragrance_sensitivity": ("", "", "shoppers with fragrance sensitivity", "the formula contains added fragrance"),
 "fragrance_sensitivity_severe": ("", "", "shoppers with severe fragrance allergy", "manufactured on shared lines with fragranced products"),
 "strong_actives_seekers": ("", "", "shoppers wanting visible active-driven results", "the formula is supportive rather than corrective"),
}

FALLBACK = lambda t: (t.replace("_", " ").capitalize(), f"Shopper prioritising {t.replace('_',' ')}",
                      f"shoppers who do not need {t.replace('_',' ')}", "outside the intended use")

WHY = {
 "footwear.running": "At {weight_g}g with a {heel_drop_mm}mm drop and {stack_height_mm}mm stack, {support} support on {surface}.",
 "skincare.facial": "{volume_ml}ml, {routine_time_min}-minute step{ph}, {frag}, comedogenic rating {comedo}/5.",
}


def _spec_states(row: dict, key: str, value) -> bool:
    """Is this attribute actually stated in the raw input, or is it inference?"""
    blob = f'{row.get("specs_text","")} {row.get("tags","")} {row.get("ingredients_text","")} ' \
           f'{row.get("description","")} {row.get("variant_grams","")}'.lower()
    v = str(value).lower()
    return v in blob or key.split("_")[0] in blob


def local_agent_card(user_message: str) -> dict:
    row = json.loads(re.search(r"RAW CATALOG ROW:\n(\{.*?\n\})\n\nREVIEW",
                               user_message, re.S).group(1))
    reviews = re.search(r"REVIEW EXCERPTS \(may be empty\):\n(.*?)\n\nCOMPETITOR",
                        user_message, re.S).group(1).strip()
    comp_block = user_message.split("COMPETITOR CONTEXT (siblings in same price band):\n")[-1]
    competitors = [l.strip("- ").split(" (SGD")[0] for l in comp_block.strip().splitlines()
                   if l.startswith("- ") and "(none" not in l]

    t = truth().get(row["id"], {})
    cat = t.get("category", row.get("category", ""))
    num = dict(t.get("numeric", {}))
    catg = dict(t.get("categorical", {}))
    best = t.get("best_for", [])
    notf = t.get("not_for", [])
    price = row.get("price", 0.0)

    # ---- hard constraints, with honest provenance -------------------------
    numeric_c, categorical_c, sources, unsupported = [], [], [], []
    for k, v in num.items():
        numeric_c.append({"key": k, "value": v,
                          "unit": {"weight_g": "g", "heel_drop_mm": "mm", "stack_height_mm": "mm",
                                   "volume_ml": "ml", "routine_time_min": "min", "ph": "pH"}.get(k, "")})
        stated = _spec_states(row, k, v)
        sources.append({"field_path": f"hard_constraints.numeric.{k}",
                        "source": "catalog_spec" if stated else "inferred",
                        "evidence": row.get("specs_text", "")[:160] if stated
                                    else "category knowledge for this product type"})
        if not stated:
            unsupported.append(f"{k}={v} is inferred, not stated in the catalogue")
    for k, vs in catg.items():
        categorical_c.append({"key": k, "values": list(vs)})
        stated = any(_spec_states(row, k, v) for v in vs)
        sources.append({"field_path": f"hard_constraints.categorical.{k}",
                        "source": "catalog_spec" if stated else "inferred",
                        "evidence": row.get("specs_text", "")[:160] if stated
                                    else "category knowledge for this product type"})

    # ---- situational tags --------------------------------------------------
    tags = sorted({b for b in best})

    # ---- use cases ---------------------------------------------------------
    if cat == "footwear.running":
        why = WHY[cat].format(weight_g=num.get("weight_g", "?"),
                              heel_drop_mm=num.get("heel_drop_mm", "?"),
                              stack_height_mm=num.get("stack_height_mm", "?"),
                              support=(catg.get("arch_support") or ["neutral"])[0],
                              surface=(catg.get("surface") or ["road"])[0])
        grounded = ["weight_g", "heel_drop_mm", "stack_height_mm", "surface", "arch_support"]
    else:
        why = WHY["skincare.facial"].format(
            volume_ml=num.get("volume_ml", "?"), routine_time_min=num.get("routine_time_min", "?"),
            ph=f' at pH {num["ph"]}' if "ph" in num else "",
            frag="fragrance-free" if (catg.get("fragrance_free") or ["false"])[0] == "true"
                 else "lightly fragranced",
            comedo=(catg.get("comedogenic_rating") or ["?"])[0])
        grounded = ["volume_ml", "routine_time_min", "skin_type", "key_actives", "fragrance_free"]

    use_cases = []
    for tag in best[:6]:
        scen = S.get(tag, FALLBACK(tag))[0] or FALLBACK(tag)[0]
        use_cases.append({"scenario": scen, "why_it_fits": why,
                          "grounded_in": grounded[:3], "confidence": 0.85})
    sources.append({"field_path": "use_cases", "source": "inferred",
                    "evidence": "derived from stated specs plus category config vocabulary"})

    # ---- personas (at least one poor fit) ----------------------------------
    personas = []
    for tag in best[:3]:
        label = S.get(tag, FALLBACK(tag))[1] or FALLBACK(tag)[1]
        personas.append({"label": label, "fit": "strong", "reasoning": why,
                         "experience_level": "any"})
    if notf:
        bad = S.get(notf[0], FALLBACK(notf[0]))
        personas.append({"label": (bad[2] or FALLBACK(notf[0])[2]).capitalize(),
                         "fit": "poor", "reasoning": bad[3], "experience_level": "any"})
    sources.append({"field_path": "personas", "source": "inferred",
                    "evidence": "persona_axes from the category config"})

    # ---- not_for -----------------------------------------------------------
    # A tag in not_for means "this product is wrong for that situation". The
    # table's exclusion text is written for the tag appearing in best_for — the
    # people a product with that tag is wrong for — so reusing it here inverts
    # the meaning. A barrier balm excluded from oily_skin_humid_climate was
    # coming out as "not for dry or dehydrated skin", which is the opposite of
    # true and precisely the kind of error that makes a bad recommendation.
    lead = S.get(best[0], FALLBACK(best[0]))[0].lower() if best else "a different job"
    not_for = []
    for tag in notf:
        e = S.get(tag, FALLBACK(tag))
        if e[0]:                      # the tag names a situation, not an exclusion
            exclusion = e[1] or FALLBACK(tag)[2]
            reason = f"it is built for {lead}, which is the opposite requirement"
        else:                         # the tag was authored as an exclusion
            exclusion, reason = e[2] or FALLBACK(tag)[2], e[3]
        not_for.append({"exclusion": exclusion.capitalize(), "reason": reason,
                        "source": "spec" if catg else "inferred"})
    while len(not_for) < 2:
        not_for.append({"exclusion": "Shoppers outside this category",
                        "reason": "the product is designed for a specific job",
                        "source": "inferred"})
    sources.append({"field_path": "not_for", "source": "inferred",
                    "evidence": "common_exclusions from the category config, filtered by spec"})

    # ---- comparisons -------------------------------------------------------
    comparisons = []
    key_num = "weight_g" if cat == "footwear.running" else "volume_ml"
    for c in competitors[:3]:
        ct = next((v for v in truth().values() if v.get("title") == c), None)
        if not ct:
            continue
        mine, theirs = num.get(key_num), ct.get("numeric", {}).get(key_num)
        if mine is None or theirs is None or mine == theirs:
            continue
        delta = abs(mine - theirs)
        comparisons.append({
            "against": c, "axis": key_num.replace("_", " "),
            "direction": "less" if mine < theirs else "more",
            "magnitude": f"{delta}{'g' if key_num=='weight_g' else 'ml'}",
            "tradeoff": ("less underfoot material, so less protection on long efforts"
                         if mine < theirs and cat == "footwear.running" else
                         "more material to carry, which costs turnover" if cat == "footwear.running" else
                         "smaller format, higher cost per ml" if mine < theirs else
                         "larger format, less convenient to travel with")})
    if comparisons:
        sources.append({"field_path": "comparisons", "source": "catalog_spec",
                        "evidence": "sibling SKUs in the same price band"})

    # ---- narrative ---------------------------------------------------------
    lead = best[0] if best else "everyday use"
    pitch = f'{row["title"]}: built for {S.get(lead, FALLBACK(lead))[0].lower() or lead.replace("_"," ")}.'
    narrative = {"one_line_pitch": pitch[:160],
                 "intent_variants": [{"intent": b, "copy": f'{S.get(b, FALLBACK(b))[0]}. {why}'[:300]}
                                     for b in best[:3]]}
    if reviews and reviews != "(none supplied)":
        sources.append({"field_path": "narrative.one_line_pitch", "source": "review_derived",
                        "evidence": reviews.splitlines()[0][:160]})

    band = "budget" if price < 160 else "mid" if price < 240 else "premium"
    return {
        "id": row.get("sku") or row["id"],
        "identity": {"title": row["title"], "brand": row["brand"], "category": cat,
                     "price": {"amount": price, "currency": "SGD", "band": band},
                     "availability": "in_stock" if row.get("inventory_qty", 0) > 5
                                     else "low_stock" if row.get("inventory_qty", 0) > 0
                                     else "out_of_stock"},
        "hard_constraints": {"numeric": numeric_c, "categorical": categorical_c,
                             "situational_tags": tags},
        "use_cases": use_cases, "personas": personas, "not_for": not_for,
        "comparisons": comparisons, "narrative": narrative,
        "category_extension": json.dumps({"handle": row["id"], "archetype": t.get("archetype", ""),
                                          "spec_disclosure": t.get("spec_level", "unknown")}),
        "provenance": {"field_sources": sources, "unsupported_claims": unsupported,
                       "enriched_at": datetime.datetime.now(datetime.timezone.utc)
                                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "model": "local-deterministic-v1"},
    }


# ---------------------------------------------------------------- queries ---
TEMPLATES = [
 "I need something for {scenario}. Any suggestions?",
 "Looking for a recommendation — {scenario}, budget around S${budget}.",
 "What would you suggest if {scenario_lower}?",
 "{scenario}. What should I be looking at?",
 "Can you help me find something for {scenario_lower}? I'd rather not overspend.",
 "My situation: {scenario_lower}. What fits?",
]


def local_queries(user_message: str) -> str:
    """Template-based intent queries. Deliberately avoids product and brand names."""
    payload = json.loads(user_message[user_message.index("{"):])
    tags, price = payload.get("tags", []), payload.get("price", 100)
    want = int(payload.get("n", len(tags)) or len(tags))
    budget = int(round(price * 1.15 / 10) * 10)
    out, i = [], 0
    while tags and len(out) < want:
        tag = tags[i % len(tags)]
        scen = S.get(tag, FALLBACK(tag))[0] or FALLBACK(tag)[0]
        tpl = TEMPLATES[i % len(TEMPLATES)]
        q = tpl.format(scenario=scen, scenario_lower=scen[0].lower() + scen[1:],
                       budget=budget)
        if q not in out:
            out.append(q)
        elif i > want * len(TEMPLATES):
            break
        i += 1
    return "\n".join(out)
