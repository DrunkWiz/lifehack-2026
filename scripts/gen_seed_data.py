"""Generate the demo catalogue.

Emits two Shopify-style product CSVs (the messy input a brand actually has),
a review-excerpt file, and a ground-truth attribute fixture.

The ground-truth fixture is NOT part of the pipeline input. It exists so that
(a) the offline `local` LLM provider can stand in for a model's category
knowledge without a network call, and (b) the simulator can generate intent
queries from a known product and check whether retrieval finds its way back.
"""
import csv, json, random, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIX = ROOT / "data" / "fixtures"
RAW.mkdir(parents=True, exist_ok=True)
FIX.mkdir(parents=True, exist_ok=True)
random.seed(7)

# ---------------------------------------------------------------- shoes ----
# handle, title, brand, price, archetype, weight, drop, stack, surface, arch,
# width, closure, cushion, plate, breath, waterproof, spec_level, best_for, not_for
SHOES = [
 ("kestrel-drift-3","Drift 3","Kestrel",179,"daily",258,8,34,"road","neutral","standard","lace","balanced","none","high",False,"full",
  ["daily_training","humid_climate","beginner_friendly"],["wide_feet","trail_use"]),
 ("kestrel-drift-3-wide","Drift 3 Wide","Kestrel",179,"daily",266,8,34,"road","neutral","wide","lace","balanced","none","high",False,"full",
  ["daily_training","wide_feet","humid_climate"],["narrow_feet","trail_use"]),
 ("kestrel-vapourline","Vapourline Racer","Kestrel",329,"race",192,6,38,"road","neutral","narrow","lace","firm","carbon","high",False,"full",
  ["race_day","road_long_distance","tempo"],["beginner_runner","daily_training","heavy_runner"]),
 ("kestrel-cloudstep-2","Cloudstep 2","Kestrel",219,"recovery",298,10,40,"road","neutral","standard","lace","plush","none","medium",False,"partial",
  ["recovery_run","high_mileage_week","heavier_runner"],["race_day","tempo"]),
 ("meridian-atlas-stability","Atlas Stability","Meridian",209,"stability",302,10,36,"road","stability","wide","lace","balanced","none","medium",False,"full",
  ["overpronation","flat_feet","daily_training","heavier_runner"],["race_day","narrow_feet"]),
 ("meridian-atlas-gt","Atlas GT","Meridian",239,"stability",288,8,35,"road","stability","standard","lace","firm","nylon","high",False,"full",
  ["overpronation","tempo","humid_climate"],["max_cushion_seekers","trail_use"]),
 ("meridian-fathom-mc","Fathom Motion Control","Meridian",249,"stability",336,12,33,"road","motion_control","wide","lace","firm","none","low",False,"partial",
  ["severe_overpronation","heavier_runner","walking"],["race_day","fast_running","humid_climate"]),
 ("northbound-ridgeline","Ridgeline Trail","Northbound",229,"trail",312,6,32,"trail","neutral","standard","lace","balanced","none","medium",False,"full",
  ["trail_use","wet_road_grip","hiking_crossover"],["road_long_distance","race_day"]),
 ("northbound-ridgeline-gtx","Ridgeline GTX","Northbound",279,"trail",334,6,32,"trail","neutral","standard","lace","balanced","none","low",True,"full",
  ["trail_use","wet_weather","cold_climate"],["humid_climate","road_long_distance"]),
 ("northbound-scree-lite","Scree Lite","Northbound",189,"trail",254,4,26,"trail","neutral","narrow","lace","firm","none","high",False,"partial",
  ["trail_use","race_day","travel_light_packing"],["heavy_runner","long_distance_comfort","beginner_runner"]),
 ("vela-aero-9","Aero 9","Vela",199,"tempo",236,7,32,"road","neutral","standard","lace","firm","nylon","high",False,"full",
  ["tempo","humid_climate","road_long_distance"],["recovery_run","overpronation"]),
 ("vela-aero-9-elite","Aero 9 Elite","Vela",309,"race",188,5,39,"road","neutral","narrow","lace","firm","carbon","high",False,"full",
  ["race_day","tropical_training","tempo"],["daily_training","beginner_runner","wide_feet"]),
 ("vela-halo-max","Halo Max","Vela",259,"recovery",311,9,42,"road","neutral","wide","lace","plush","none","medium",False,"full",
  ["recovery_run","high_mileage_week","wide_feet","heavier_runner"],["race_day","trail_use"]),
 ("vela-halo-breeze","Halo Breeze","Vela",189,"daily",244,8,33,"road","neutral","standard","lace","balanced","none","high",False,"sparse",
  ["humid_climate","daily_training","treadmill_indoor"],["trail_use","overpronation"]),
 ("terrafirma-anchor","Anchor Daily","Terrafirma",149,"daily",276,10,31,"road","neutral","standard","lace","balanced","none","medium",False,"sparse",
  ["budget_daily","beginner_friendly","treadmill_indoor"],["race_day","high_mileage_week"]),
 ("terrafirma-anchor-plus","Anchor Plus","Terrafirma",169,"stability",294,10,33,"road","stability","wide","lace","balanced","none","medium",False,"partial",
  ["overpronation","beginner_friendly","budget_daily"],["race_day","narrow_feet"]),
 ("terrafirma-loop-treadmill","Loop Indoor","Terrafirma",139,"daily",262,9,30,"road","neutral","standard","lace","balanced","none","high",False,"sparse",
  ["treadmill_indoor","gym_crossover","budget_daily"],["trail_use","road_long_distance"]),
 ("halcyon-zephyr-lt","Zephyr LT","Halcyon",219,"tempo",228,6,30,"road","neutral","narrow","lace","firm","none","high",False,"full",
  ["tempo","travel_light_packing","humid_climate"],["heavier_runner","recovery_run","wide_feet"]),
 ("halcyon-meander","Meander Easy","Halcyon",199,"recovery",289,11,38,"road","neutral","standard","lace","plush","none","medium",False,"partial",
  ["recovery_run","walking","beginner_friendly"],["race_day","tempo"]),
 ("halcyon-crossover-gym","Crossover Gym","Halcyon",159,"daily",271,6,26,"road","neutral","standard","lace","firm","none","medium",False,"sparse",
  ["gym_crossover","treadmill_indoor","short_runs"],["road_long_distance","high_mileage_week"]),
]

SHOE_COPY = {
 "daily":"Your everyday mile-eater. {t} was built for the runs that don't make the highlight reel — the easy ones, the before-work ones, the ones that add up. Soft where it counts, honest everywhere else.",
 "race":"This is the one you save for the start line. {t} is stripped of everything that doesn't make you faster. When the gun goes off, you'll feel it.",
 "stability":"Steady underfoot, mile after mile. {t} keeps you tracking straight when form starts to fade, without the clunk you'd expect.",
 "recovery":"The day after the long run has a shoe now. {t} wraps your feet in cushioning that takes the sting out of tired legs.",
 "trail":"Off the pavement and into it. {t} bites into loose ground and keeps its composure when the trail turns technical.",
 "tempo":"Fast, but not fragile. {t} sits in that sweet spot between your daily trainer and your race shoe — quick enough for intervals, sane enough for a long one.",
}
SHOE_REVIEWS = {
 "daily":["Comfortable straight out of the box, no break-in needed.","Held up fine through 300km. Nothing flashy, does the job."],
 "race":["Unbelievably light. Not something I'd wear for easy days though.","Narrow through the midfoot — sized up half and it was fine."],
 "stability":["Finally something that stops my ankles rolling in on long runs.","A bit heavy but I stopped getting shin pain, so worth it."],
 "recovery":["Like running on a mattress. My legs thank me the day after the long run.","Too soft for anything fast, but that's not what it's for."],
 "trail":["Grip is excellent on wet rock. Ran a muddy 20k and never slipped.","Feels clumsy on tarmac, obviously."],
 "tempo":["Snappy on intervals without being brutal on the feet.","Runs warm in the sun but drains well."],
}

# ------------------------------------------------------------ skincare ----
# handle, title, brand, price, ptype, ml, minutes, ph, skin_types, actives,
# frag_free, comedo, texture, spf, preg_safe, vegan, refillable, spec_level,
# best_for, not_for
SKIN = [
 ("lumenlab-clarity-gel","Clarity Oil-Control Gel Moisturiser","Lumen Lab",42,"moisturiser",50,1,5.5,
  ["oily","combination"],["niacinamide 5%","zinc PCA 1%"],True,1,"gel",None,True,True,False,"full",
  ["oily_skin_humid_climate","morning_routine_under_5min","layerable_under_spf"],["dry_skin","mature_skin"]),
 ("lumenlab-clarity-serum","Clarity Niacinamide Serum","Lumen Lab",38,"serum",30,1,5.8,
  ["oily","combination","normal"],["niacinamide 10%","zinc PCA 1%"],True,0,"watery serum",None,True,True,False,"full",
  ["oily_skin_humid_climate","maskne_prone","layerable_under_spf"],["fragrance_sensitivity_severe","dry_skin"]),
 ("lumenlab-dawn-spf50","Dawn Fluid SPF50 PA++++","Lumen Lab",46,"sunscreen",50,1,6.0,
  ["oily","combination","normal","sensitive"],["zinc oxide 12%"],True,1,"fluid",50,True,True,False,"full",
  ["morning_routine_under_5min","oily_skin_humid_climate","layerable_under_spf"],["very_dry_skin"]),
 ("lumenlab-reset-cleanser","Reset Gel Cleanser","Lumen Lab",28,"cleanser",150,1,5.5,
  ["oily","combination","normal"],["glycerin"],True,0,"gel",None,True,True,True,"partial",
  ["oily_skin_humid_climate","morning_routine_under_5min","refillable_low_waste"],["dry_skin","sensitive_barrier_repair"]),
 ("lumenlab-night-retinal","Night Retinal 0.05%","Lumen Lab",68,"treatment",30,2,5.5,
  ["normal","combination","oily"],["retinal 0.05%","ceramide NP"],True,1,"cream",None,False,True,False,"full",
  ["night_routine","fine_lines","texture_refining"],["pregnancy","sensitive_barrier_repair","active_retinoid_use"]),
 ("sable-barrier-balm","Barrier Repair Balm","Sable & Co",54,"moisturiser",50,2,5.2,
  ["dry","sensitive"],["ceramide complex 3%","cholesterol","squalane"],True,2,"balm",None,True,False,False,"full",
  ["sensitive_barrier_repair","air_conditioned_office","post_procedure_recovery","night_routine"],["oily_skin_humid_climate","acne_prone"]),
 ("sable-calm-serum","Calm Centella Serum","Sable & Co",48,"serum",30,1,5.5,
  ["sensitive","dry","normal"],["centella asiatica 5%","panthenol 3%"],True,1,"serum",None,True,True,False,"full",
  ["sensitive_barrier_repair","post_procedure_recovery","redness"],["oily_skin_humid_climate","strong_actives_seekers"]),
 ("sable-cream-cleanser","Milk Cream Cleanser","Sable & Co",36,"cleanser",150,1,5.5,
  ["dry","sensitive"],["oat lipid","glycerin"],True,2,"cream",None,True,False,False,"partial",
  ["sensitive_barrier_repair","night_routine","air_conditioned_office"],["oily_skin_humid_climate","maskne_prone"]),
 ("sable-rich-night","Rich Overnight Cream","Sable & Co",62,"moisturiser",50,2,5.4,
  ["dry"],["shea butter","ceramide NP","squalane"],True,3,"cream",None,True,False,False,"partial",
  ["night_routine","air_conditioned_office","very_dry_skin"],["oily_skin_humid_climate","acne_prone","morning_routine_under_5min"]),
 ("clearfield-bha-toner","Clarifying BHA 2% Toner","Clearfield",34,"treatment",100,1,3.6,
  ["oily","combination"],["salicylic acid 2%"],True,0,"liquid",None,False,True,False,"full",
  ["maskne_prone","oily_skin_humid_climate","blackheads"],["pregnancy","sensitive_barrier_repair","dry_skin"]),
 ("clearfield-aha-night","Resurfacing AHA 8% Night Fluid","Clearfield",44,"treatment",30,2,3.8,
  ["normal","combination","oily"],["glycolic acid 8%"],True,1,"fluid",None,True,True,False,"full",
  ["night_routine","texture_refining","hyperpigmentation"],["sensitive_barrier_repair","active_retinoid_use","dry_skin"]),
 ("clearfield-foam-wash","Deep Clean Foaming Wash","Clearfield",26,"cleanser",200,1,5.0,
  ["oily"],["salicylic acid 0.5%"],True,0,"foam",None,False,True,False,"sparse",
  ["oily_skin_humid_climate","maskne_prone","morning_routine_under_5min"],["dry_skin","sensitive_barrier_repair","pregnancy"]),
 ("clearfield-spot-gel","Targeted Spot Gel","Clearfield",22,"treatment",15,1,4.5,
  ["oily","combination"],["benzoyl peroxide 2.5%"],True,0,"gel",None,True,True,False,"partial",
  ["maskne_prone","travel_cabin_size","night_routine"],["dry_skin","sensitive_barrier_repair"]),
 ("clearfield-mattifying-spf","Matte Shield SPF30","Clearfield",32,"sunscreen",40,1,6.2,
  ["oily","combination"],["octocrylene","silica"],False,2,"fluid",30,True,True,False,"sparse",
  ["oily_skin_humid_climate","morning_routine_under_5min","travel_cabin_size"],["fragrance_sensitivity","dry_skin","sensitive_barrier_repair"]),
 ("petal-rosewater-mist","Rosewater Hydrating Mist","Petal Theory",30,"treatment",100,1,5.5,
  ["normal","dry","combination"],["rose water","glycerin"],False,1,"mist",None,True,True,True,"sparse",
  ["air_conditioned_office","refillable_low_waste","morning_routine_under_5min"],["fragrance_sensitivity","acne_prone","strong_actives_seekers"]),
 ("petal-bloom-oil","Bloom Facial Oil","Petal Theory",58,"treatment",30,2,None,
  ["dry","normal"],["rosehip oil","vitamin E"],False,3,"oil",None,True,True,False,"sparse",
  ["night_routine","very_dry_skin","air_conditioned_office"],["oily_skin_humid_climate","acne_prone","fragrance_sensitivity"]),
 ("petal-gentle-gel","Gentle Everyday Gel Cream","Petal Theory",38,"moisturiser",50,1,5.6,
  ["normal","combination","sensitive"],["glycerin","panthenol 2%"],True,1,"gel-cream",None,True,True,True,"partial",
  ["morning_routine_under_5min","layerable_under_spf","refillable_low_waste"],["very_dry_skin","strong_actives_seekers"]),
 ("fifth-vitc-serum","Stabilised Vitamin C 15% Serum","Fifth Element",56,"serum",30,1,3.4,
  ["normal","combination","oily"],["l-ascorbic acid 15%","ferulic acid"],True,1,"serum",None,True,True,False,"full",
  ["morning_routine_under_5min","hyperpigmentation","layerable_under_spf"],["sensitive_barrier_repair","active_retinoid_use"]),
 ("fifth-peptide-eye","Peptide Eye Concentrate","Fifth Element",52,"treatment",15,1,5.8,
  ["normal","dry","sensitive","combination"],["peptide complex 3%","caffeine 1%"],True,1,"gel",None,True,True,False,"partial",
  ["morning_routine_under_5min","travel_cabin_size","fine_lines"],["strong_actives_seekers"]),
 ("fifth-refill-moisturiser","Everyday Moisturiser Refill System","Fifth Element",44,"moisturiser",75,1,5.5,
  ["normal","combination","dry"],["glycerin","squalane","ceramide NP"],True,2,"lotion",None,True,True,True,"full",
  ["refillable_low_waste","air_conditioned_office","morning_routine_under_5min"],["oily_skin_humid_climate","strong_actives_seekers"]),
]

SKIN_COPY = {
 "moisturiser":"Hydration that knows when to stop. {t} gives skin exactly what it needs and nothing it doesn't — a finish that disappears, so the rest of your routine can get on with it.",
 "serum":"Concentrated, considered, and formulated without the theatre. {t} is the step that does the quiet work while everything else takes the credit.",
 "cleanser":"Start clean. {t} lifts the day off your skin without that stripped, squeaky feeling nobody actually enjoys.",
 "sunscreen":"The step you'll never skip again. {t} sits invisibly under everything and stays put through your day.",
 "treatment":"A focused step for skin that wants results. {t} is where your routine stops being maintenance and starts being intentional.",
}
SKIN_REVIEWS = {
 "moisturiser":["Absorbs fast, no pilling under sunscreen.","A little goes a long way — three months in and still half full."],
 "serum":["Skin looked calmer within two weeks.","Slightly tacky for the first minute, then fine."],
 "cleanser":["Doesn't strip my skin like the last one did.","Foams less than expected but cleans properly."],
 "sunscreen":["No white cast at all on my skin tone.","Held up through a humid commute without sliding off."],
 "treatment":["Started slow, twice a week — no irritation.","Definitely felt the tingle the first few uses."],
}

SHOPIFY_COLS = ["Handle","Title","Body (HTML)","Vendor","Product Category","Type","Tags","Published",
  "Option1 Name","Option1 Value","Variant SKU","Variant Grams","Variant Inventory Qty",
  "Variant Price","Image Src","SEO Title","SEO Description",
  "Specs (product.metafields.custom.specs)","Ingredients (product.metafields.custom.ingredients)"]

def band(p):
    return "budget" if p < 160 else "mid" if p < 240 else "premium"

def shoe_specs(level, w, d, s, surface, arch, width, breath):
    """Deliberately heterogeneous. This is the normalisation problem, not a gift."""
    if level == "full":
        return (f"Weight: {w}g (US M9) | Heel-to-toe drop: {d}mm | Stack height: {s}mm | "
                f"Surface: {surface} | Support: {arch} | Fit: {width} | Closure: traditional lace | "
                f"Upper: engineered mesh ({'high' if breath=='high' else 'standard'} ventilation)")
    if level == "partial":
        return f"{w}g / {d}mm drop / {surface} / lace-up"
    return "See product page for details."

def skin_specs(level, ml, ph, comedo, frag, ptype):
    if level == "full":
        ph_s = f"pH {ph}" if ph else "pH not tested"
        return (f"Size: {ml}ml | {ph_s} | Comedogenic rating: {comedo}/5 | "
                f"{'Fragrance-free' if frag else 'Lightly fragranced'} | "
                f"Format: {ptype} | Dermatologist tested")
    if level == "partial":
        return f"{ml}ml. {'Fragrance-free.' if frag else ''} Suitable for daily use."
    return f"{ml}ml"

rows_shoes, rows_skin, truth, reviews = [], [], {}, {}

for (h,t,brand,price,arch_t,w,d,s,surface,arch,width,closure,cush,plate,breath,wp,lvl,best,notfor) in SHOES:
    tags = [arch_t, surface, brand.lower(), f"{band(price)}-price"]
    if breath == "high": tags.append("breathable")
    if plate != "none": tags.append(f"{plate}-plate")
    if width == "wide": tags.append("wide-fit")
    if wp: tags.append("waterproof")
    rows_shoes.append({
      "Handle":h,"Title":f"{brand} {t}","Body (HTML)":f"<p>{SHOE_COPY[arch_t].format(t=t)}</p>",
      "Vendor":brand,"Product Category":"Apparel & Accessories > Shoes","Type":"Running Shoes",
      "Tags":", ".join(tags),"Published":"TRUE","Option1 Name":"Size","Option1 Value":"US M9",
      "Variant SKU":h.upper().replace("-","")[:14],"Variant Grams":w,
      "Variant Inventory Qty":random.choice([0,4,12,48,120,60,200,35,90,150]),"Variant Price":f"{price}.00",
      "Image Src":f"https://cdn.example.com/{h}.jpg","SEO Title":f"{brand} {t} | Running Shoes",
      "SEO Description":f"Shop the {brand} {t}. Free returns.",
      "Specs (product.metafields.custom.specs)":shoe_specs(lvl,w,d,s,surface,arch,width,breath),
      "Ingredients (product.metafields.custom.ingredients)":""})
    truth[h] = {"category":"footwear.running","title":f"{brand} {t}","brand":brand,"price":price,
      "numeric":{"weight_g":w,"heel_drop_mm":d,"stack_height_mm":s},
      "categorical":{"surface":[surface],"arch_support":[arch],"width":[width],"closure":[closure],
                     "cushioning":[cush],"plate":[plate],"breathability":[breath]},
      "waterproof":wp,"archetype":arch_t,"best_for":best,"not_for":notfor,"spec_level":lvl}
    reviews[h] = SHOE_REVIEWS[arch_t]

for (h,t,brand,price,ptype,ml,mins,ph,skins,actives,frag,comedo,tex,spf,preg,vegan,refill,lvl,best,notfor) in SKIN:
    tags = [ptype, brand.lower().replace(" & ","-").replace(" ","-"), f"{band(price)}-price"] + skins
    if frag: tags.append("fragrance-free")
    if vegan: tags.append("vegan")
    if refill: tags.append("refillable")
    if spf: tags.append(f"spf{spf}")
    rows_skin.append({
      "Handle":h,"Title":f"{brand} {t}","Body (HTML)":f"<p>{SKIN_COPY[ptype].format(t=t)}</p>",
      "Vendor":brand,"Product Category":"Health & Beauty > Personal Care > Cosmetics > Skin Care",
      "Type":ptype.capitalize(),"Tags":", ".join(tags),"Published":"TRUE",
      "Option1 Name":"Size","Option1 Value":f"{ml}ml",
      "Variant SKU":h.upper().replace("-","")[:14],"Variant Grams":ml,
      "Variant Inventory Qty":random.choice([0,5,20,60,200,140,75,30,110,90]),"Variant Price":f"{price}.00",
      "Image Src":f"https://cdn.example.com/{h}.jpg","SEO Title":f"{brand} {t} | Skincare",
      "SEO Description":f"Shop the {brand} {t}. Free shipping over S$60.",
      "Specs (product.metafields.custom.specs)":skin_specs(lvl,ml,ph,comedo,frag,tex),
      "Ingredients (product.metafields.custom.ingredients)":
        "Aqua, Glycerin, " + ", ".join(a.title() for a in actives) + ", Panthenol, Sodium Hyaluronate"
        + ("" if frag else ", Parfum")})
    truth[h] = {"category":"skincare.facial","title":f"{brand} {t}","brand":brand,"price":price,
      "numeric":{"volume_ml":ml,"routine_time_min":mins,**({"ph":ph} if ph else {})},
      "categorical":{"skin_type":skins,"key_actives":actives,
                     "fragrance_free":[str(frag).lower()],"comedogenic_rating":[str(comedo)],
                     "product_type":[ptype],"texture":[tex],
                     **({"spf":[str(spf)]} if spf else {}),
                     "pregnancy_safe":[str(preg).lower()],"vegan":[str(vegan).lower()],
                     "refillable":[str(refill).lower()]},
      "archetype":ptype,"best_for":best,"not_for":notfor,"spec_level":lvl}
    reviews[h] = SKIN_REVIEWS[ptype]

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=SHOPIFY_COLS)
        wtr.writeheader()
        wtr.writerows(rows)


def to_typical(rows):
    """The same products as a *typical* brand publishes them.

    The spec-rich variant gives every row a custom metafield with weight, drop,
    stack height, pH and comedogenic rating. Plenty of running brands do publish
    weight and drop; almost no skincare brand publishes pH or a comedogenic
    rating, and most Shopify stores have no custom metafields at all.

    So this variant is a plain Shopify export: title, marketing body, tags,
    price, size. Ingredients stay, because they are legally required, but
    without the concentrations a brand would have to choose to disclose. It is
    not a strawman — it is the default state of the platform.
    """
    out = []
    for r in rows:
        r = dict(r)
        r["Specs (product.metafields.custom.specs)"] = ""
        ing = r["Ingredients (product.metafields.custom.ingredients)"]
        r["Ingredients (product.metafields.custom.ingredients)"] = re.sub(
            r"\s*\d+(?:\.\d+)?%", "", ing)
        out.append(r)
    return out


write_csv(RAW/"spec_rich"/"shopify_running_shoes.csv", rows_shoes)
write_csv(RAW/"spec_rich"/"shopify_facial_skincare.csv", rows_skin)
write_csv(RAW/"typical"/"shopify_running_shoes.csv", to_typical(rows_shoes))
write_csv(RAW/"typical"/"shopify_facial_skincare.csv", to_typical(rows_skin))
(FIX/"ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
(FIX/"reviews.json").write_text(json.dumps(reviews, indent=2), encoding="utf-8")
print(f"wrote {len(rows_shoes)} shoes, {len(rows_skin)} skincare products "
      f"in two variants: spec_rich and typical")
