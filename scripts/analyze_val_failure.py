#!/usr/bin/env python3
"""
Analyze WHY no validation scheme tracks the 2024 test metric.
Explores: hybrid novelty, environment novelty, year structure, GxE interactions.
"""

import pandas as pd
import numpy as np
from collections import Counter

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
X_train = pd.read_csv("data/maize_data_2014-2023_vs_2024_v2/X_train.csv", usecols=["id", "Env"], dtype=str)
X_test  = pd.read_csv("data/maize_data_2014-2023_vs_2024_v2/X_test.csv",  usecols=["id", "Env"], dtype=str)
y_train = pd.read_csv("data/maize_data_2014-2023_vs_2024_v2/y_train.csv")
y_test  = pd.read_csv("data/maize_data_2014-2023_vs_2024_v2/y_test.csv")

# Ensure yield is numeric
y_train["Yield_Mg_ha"] = pd.to_numeric(y_train["Yield_Mg_ha"], errors="coerce")
y_test["Yield_Mg_ha"] = pd.to_numeric(y_test["Yield_Mg_ha"], errors="coerce")

# Parse hybrid and environment from id
# id format: "ENV-HYBRID" e.g. "DEH1_2014-M0088/LH185"
X_train["hybrid"] = X_train["id"].str.split("-", n=1).str[1]
X_test["hybrid"]  = X_test["id"].str.split("-", n=1).str[1]

# Parse year from Env
X_train["year"] = X_train["Env"].str.extract(r"(\d{4})").astype(int)
X_test["year"]  = X_test["Env"].str.extract(r"(\d{4})").astype(int)

# Parse location (everything before the year)
X_train["location"] = X_train["Env"].str.replace(r"_\d{4}$", "", regex=True)
X_test["location"]  = X_test["Env"].str.replace(r"_\d{4}$", "", regex=True)

X_train["yield"] = y_train["Yield_Mg_ha"].values
X_test["yield"]  = y_test["Yield_Mg_ha"].values

print(f"Train: {len(X_train):,} rows, Test: {len(X_test):,} rows")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 1. HYBRID NOVELTY
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. HYBRID NOVELTY ANALYSIS")
print("=" * 70)
train_hybrids = set(X_train["hybrid"].unique())
test_hybrids  = set(X_test["hybrid"].unique())
novel_hybrids = test_hybrids - train_hybrids
shared_hybrids = test_hybrids & train_hybrids

print(f"Train hybrids: {len(train_hybrids):,}")
print(f"Test hybrids:  {len(test_hybrids):,}")
print(f"Shared:        {len(shared_hybrids):,} ({100*len(shared_hybrids)/len(test_hybrids):.1f}% of test)")
print(f"Novel in test: {len(novel_hybrids):,} ({100*len(novel_hybrids)/len(test_hybrids):.1f}% of test)")

# How many TEST ROWS have novel hybrids?
test_novel_mask = X_test["hybrid"].isin(novel_hybrids)
print(f"\nTest rows with novel hybrids: {test_novel_mask.sum():,} / {len(X_test):,} ({100*test_novel_mask.mean():.1f}%)")
test_shared_mask = ~test_novel_mask
print(f"Test rows with seen hybrids:  {test_shared_mask.sum():,} / {len(X_test):,} ({100*test_shared_mask.mean():.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. ENVIRONMENT NOVELTY
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("2. ENVIRONMENT NOVELTY ANALYSIS")
print("=" * 70)
train_envs = set(X_train["Env"].unique())
test_envs  = set(X_test["Env"].unique())
train_locs = set(X_train["location"].unique())
test_locs  = set(X_test["location"].unique())

print(f"Train environments: {len(train_envs)} across years {sorted(X_train['year'].unique())}")
print(f"Test environments:  {len(test_envs)} (all 2024)")
print(f"\nTrain locations: {len(train_locs)}")
print(f"Test locations:  {len(test_locs)}")
print(f"Shared locations: {len(train_locs & test_locs)}")
print(f"Novel test locations: {len(test_locs - train_locs)}")
print(f"  Novel: {test_locs - train_locs}")

# How many years does each test location appear in training?
print("\nTest location coverage in training years:")
for loc in sorted(test_locs):
    years = sorted(X_train[X_train["location"] == loc]["year"].unique())
    print(f"  {loc}: appeared in {len(years)} years: {years}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. YEAR-OVER-YEAR HYBRID TURNOVER
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("3. YEAR-OVER-YEAR HYBRID TURNOVER")
print("=" * 70)

years = sorted(X_train["year"].unique())
all_years = years + [2024]

# For each consecutive pair of years, how many hybrids are shared?
for i in range(len(all_years) - 1):
    y1, y2 = all_years[i], all_years[i+1]
    if y1 == 2024:
        h1 = set(X_test["hybrid"].unique())
    else:
        h1 = set(X_train[X_train["year"] == y1]["hybrid"].unique())
    if y2 == 2024:
        h2 = set(X_test["hybrid"].unique())
    else:
        h2 = set(X_train[X_train["year"] == y2]["hybrid"].unique())
    shared = len(h1 & h2)
    print(f"  {y1}→{y2}: {len(h1)} → {len(h2)} hybrids, {shared} shared ({100*shared/len(h2):.1f}% of {y2})")

# Cumulative: how many 2024 hybrids appeared in ANY prior year?
print(f"\nCumulative novel hybrid rate by holdout year:")
for holdout_yr in all_years[1:]:
    if holdout_yr == 2024:
        holdout_h = set(X_test["hybrid"].unique())
        prior_h = set(X_train["hybrid"].unique())
    else:
        holdout_h = set(X_train[X_train["year"] == holdout_yr]["hybrid"].unique())
        prior_h = set(X_train[X_train["year"] < holdout_yr]["hybrid"].unique())
    novel = len(holdout_h - prior_h)
    print(f"  Hold out {holdout_yr}: {novel}/{len(holdout_h)} novel ({100*novel/len(holdout_h):.1f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# 4. THE CORE PROBLEM: GxE COMPOSITION BY YEAR
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("4. HYBRID × LOCATION OVERLAP WITH TEST BY YEAR")
print("=" * 70)

# For each training year, what fraction of (hybrid, location) pairs overlap with test?
test_pairs = set(zip(X_test["hybrid"], X_test["location"]))
test_hybrids_set = set(X_test["hybrid"])
test_locs_set = set(X_test["location"])

for yr in all_years[:-1]:
    yr_data = X_train[X_train["year"] == yr]
    yr_hybrids = set(yr_data["hybrid"].unique())
    yr_locs = set(yr_data["location"].unique())
    yr_pairs = set(zip(yr_data["hybrid"], yr_data["location"]))
    
    h_overlap = len(yr_hybrids & test_hybrids_set)
    l_overlap = len(yr_locs & test_locs_set)
    
    print(f"  {yr}: {len(yr_data):>5} rows, {len(yr_hybrids):>4} hybrids ({h_overlap} shared w/ test), "
          f"{len(yr_locs):>3} locs ({l_overlap} shared w/ test)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. WHY LEO FAILS: environment-level yield variance
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("5. WHY LEO FAILS: ENV YIELD DISTRIBUTIONS")
print("=" * 70)

# LEO holds out entire environments. But what matters for the test metric
# is per-environment Pearson correlation (ranking within env).
# LEO tests spatial interpolation: "can you predict yield at a new location?"
# Test 2024 tests temporal extrapolation: "can you predict yield in a new year?"

# The key question: does performance on held-out envs from 2014-2023 predict
# performance on 2024 envs?

# Yield statistics by year
print("Mean yield by year:")
for yr in all_years:
    if yr == 2024:
        yields = X_test["yield"]
    else:
        yields = X_train[X_train["year"] == yr]["yield"]
    print(f"  {yr}: mean={yields.mean():.2f}, std={yields.std():.2f}, n={len(yields)}")

# Within-environment variance (what env_avg_pearson cares about)
print("\nWithin-environment yield std by year:")
for yr in all_years:
    if yr == 2024:
        data = X_test
    else:
        data = X_train[X_train["year"] == yr]
    env_stds = data.groupby("Env")["yield"].std().dropna()
    print(f"  {yr}: median_within_env_std={env_stds.median():.2f}, "
          f"mean={env_stds.mean():.2f}, n_envs={len(env_stds)}")

# ═══════════════════════════════════════════════════════════════════════════
# 6. KEY: HYBRID LIFECYCLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("6. HYBRID LIFECYCLE: HOW LONG DO HYBRIDS PERSIST?")
print("=" * 70)

# For all train hybrids, how many years do they appear?
hybrid_years = X_train.groupby("hybrid")["year"].apply(lambda x: sorted(x.unique()))
hybrid_lifespans = hybrid_years.apply(len)
print(f"Hybrid lifespan distribution (in training 2014-2023):")
for span in sorted(hybrid_lifespans.unique()):
    count = (hybrid_lifespans == span).sum()
    print(f"  {span} year(s): {count} hybrids ({100*count/len(hybrid_lifespans):.1f}%)")

# For test hybrids that ARE in training, when did they last appear?
shared_test_hybrids = test_hybrids_set & train_hybrids
if shared_test_hybrids:
    print(f"\nFor the {len(shared_test_hybrids)} shared hybrids, last training year:")
    last_years = {}
    for h in shared_test_hybrids:
        last_yr = X_train[X_train["hybrid"] == h]["year"].max()
        last_years[h] = last_yr
    last_yr_counts = Counter(last_years.values())
    for yr in sorted(last_yr_counts.keys()):
        print(f"  Last seen {yr}: {last_yr_counts[yr]} hybrids")

# ═══════════════════════════════════════════════════════════════════════════
# 7. SIMULATING WHAT EACH VAL SCHEME ACTUALLY TESTS
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("7. CHARACTERIZING EACH VALIDATION SCHEME vs 2024 TEST")
print("=" * 70)

print("\nTEST 2024 composition:")
print(f"  - {100*len(novel_hybrids)/len(test_hybrids_set):.0f}% novel hybrids")
print(f"  - 100% novel environments (all _2024)")
print(f"  - Requires: extrapolation to new hybrids × new year-specific weather")

print("\nLEO (Leave-Environment-Out from 2014-2023):")
# When you hold out an env, say IAH2_2020, you predict yield for hybrids that
# were also grown at other locations in 2020 or other years.
# The held-out envs have the SAME hybrids in training (at other locations).
# But 2024 test has 90% NOVEL hybrids.
print(f"  - Held-out envs contain hybrids seen at other locs in same/other years")
print(f"  - Tests SPATIAL generalization with KNOWN hybrids")
print(f"  - 2024 test requires TEMPORAL+GENETIC generalization with NOVEL hybrids")
print(f"  → Mismatch: LEO never tests novel-hybrid prediction")

print("\nLGO (Leave-Genotype-Out):")
print(f"  - Holds out specific hybrids from ALL years/envs")
print(f"  - Tests GENETIC generalization with KNOWN environments")
print(f"  - But 2024 test also has completely new environments (weather/year)")
print(f"  → Partial match on hybrid novelty, misses environmental novelty")

print("\nLYO (Leave-Year-Out, e.g., hold out 2023):")
print(f"  - Holds out one year, retains all others")
print(f"  - Tests TEMPORAL generalization")
# Key: what's the hybrid overlap between 2023 held-out and its training set?
if 2023 in X_train["year"].values:
    h_2023 = set(X_train[X_train["year"] == 2023]["hybrid"].unique())
    h_before_2023 = set(X_train[X_train["year"] < 2023]["hybrid"].unique())
    h_2024_test = test_hybrids_set
    h_before_2024 = train_hybrids
    
    novel_2023 = len(h_2023 - h_before_2023)
    novel_2024 = len(h_2024_test - h_before_2024)
    
    print(f"  - 2023 holdout: {novel_2023}/{len(h_2023)} novel hybrids ({100*novel_2023/len(h_2023):.1f}%)")
    print(f"  - 2024 test:    {novel_2024}/{len(h_2024_test)} novel hybrids ({100*novel_2024/len(h_2024_test):.1f}%)")
    print(f"  → If novelty rates differ hugely, LYO doesn't replicate the test challenge")

# ═══════════════════════════════════════════════════════════════════════════
# 8. PROPOSED SOLUTION: SIMULATED NOVEL-HYBRID + NOVEL-YEAR VAL
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("8. PROPOSED SOLUTION ANALYSIS")  
print("=" * 70)

# The test is: predict yield for (mostly novel hybrids) × (novel year/weather).
# No single existing scheme replicates this. We need a scheme that simultaneously:
#   (a) holds out HYBRIDS the model has never seen  
#   (b) holds out an entire YEAR the model has never seen
#   (c) only evaluates on (novel hybrid × novel year) combos
#
# This is Leave-Year-AND-Genotype-Out (LYGO).
# Hold out year Y, AND from year Y, only evaluate on hybrids not seen in prior years.

print("PROPOSED: Leave-Year-and-Novel-Genotype-Out (LYNGO)")
print("  For each holdout year Y:")
print("    - Train on all data from years < Y")
print("    - From year Y, identify hybrids NOT in years < Y")
print("    - Validate ONLY on novel hybrids in year Y")
print("    - Metric: env_avg_pearson on those rows")
print()

# Simulate this for each year
print("Simulated LYNGO validation set sizes:")
for yr in range(2016, 2024):  # need at least 2 years of training
    train_data = X_train[X_train["year"] < yr]
    val_data = X_train[X_train["year"] == yr]
    
    train_h = set(train_data["hybrid"].unique())
    val_h = set(val_data["hybrid"].unique())
    novel_h = val_h - train_h
    
    # Filter val to only novel hybrids
    val_novel = val_data[val_data["hybrid"].isin(novel_h)]
    val_envs = val_novel["Env"].nunique()
    
    # For env_avg_pearson, need >= 2 samples per env
    env_counts = val_novel.groupby("Env").size()
    usable_envs = (env_counts >= 10).sum()  # need enough for meaningful correlation
    
    print(f"  Hold out {yr}: {len(val_novel):>5} novel-hybrid rows across {val_envs} envs "
          f"({usable_envs} with >=10 samples), novel rate={100*len(novel_h)/len(val_h):.0f}%")

# Compare with actual 2024 test
print(f"\n  Actual 2024 test: {len(X_test):>5} rows across {X_test['Env'].nunique()} envs, "
      f"novel rate={100*len(novel_hybrids)/len(test_hybrids_set):.0f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 9. PARENT-LEVEL ANALYSIS (hybrid = parent1/parent2)
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("9. PARENT-LEVEL NOVELTY (hybrids = P1/P2 crosses)")
print("=" * 70)

# Parse parents from hybrid names
def parse_parents(hybrid):
    if "/" in str(hybrid):
        parts = str(hybrid).split("/")
        return parts[0], parts[1] if len(parts) > 1 else (parts[0], None)
    return (str(hybrid), None)

train_parents = X_train["hybrid"].apply(parse_parents)
test_parents = X_test["hybrid"].apply(parse_parents)

X_train["parent1"] = train_parents.apply(lambda x: x[0])
X_train["parent2"] = train_parents.apply(lambda x: x[1])
X_test["parent1"] = test_parents.apply(lambda x: x[0])
X_test["parent2"] = test_parents.apply(lambda x: x[1])

train_p1 = set(X_train["parent1"].dropna().unique())
train_p2 = set(X_train["parent2"].dropna().unique())
test_p1 = set(X_test["parent1"].dropna().unique())
test_p2 = set(X_test["parent2"].dropna().unique())

train_all_parents = train_p1 | train_p2
test_all_parents = test_p1 | test_p2

print(f"Train parents: {len(train_all_parents)}")
print(f"Test parents:  {len(test_all_parents)}")
print(f"Shared parents: {len(train_all_parents & test_all_parents)}")
print(f"Novel test parents: {len(test_all_parents - train_all_parents)}")

# For novel hybrids, how many have at least one known parent?
novel_hybrid_data = X_test[X_test["hybrid"].isin(novel_hybrids)]
has_known_p1 = novel_hybrid_data["parent1"].isin(train_all_parents).mean()
has_known_p2 = novel_hybrid_data["parent2"].isin(train_all_parents).mean()
has_any_known = (novel_hybrid_data["parent1"].isin(train_all_parents) | 
                 novel_hybrid_data["parent2"].isin(train_all_parents)).mean()
has_both_known = (novel_hybrid_data["parent1"].isin(train_all_parents) & 
                  novel_hybrid_data["parent2"].isin(train_all_parents)).mean()

print(f"\nAmong NOVEL test hybrids ({len(novel_hybrids)}):")
print(f"  Has known parent1: {100*has_known_p1:.1f}%")
print(f"  Has known parent2: {100*has_known_p2:.1f}%")
print(f"  Has ANY known parent: {100*has_any_known:.1f}%")
print(f"  Has BOTH known parents: {100*has_both_known:.1f}%")

# What are the most common test parents?
print(f"\nTop 10 test parent1 lines:")
p1_counts = X_test["parent1"].value_counts().head(10)
for p, c in p1_counts.items():
    in_train = "KNOWN" if p in train_all_parents else "NOVEL"
    print(f"  {p}: {c} rows ({in_train})")

print(f"\nTop 10 test parent2 lines:")
p2_counts = X_test["parent2"].value_counts().head(10)
for p, c in p2_counts.items():
    in_train = "KNOWN" if p in train_all_parents else "NOVEL"
    print(f"  {p}: {c} rows ({in_train})")

# ═══════════════════════════════════════════════════════════════════════════
# 10. YEAR-SPECIFIC ANALYSIS: NOVELTY RATE TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("10. NOVELTY TRAJECTORY OVER YEARS")
print("=" * 70)

# For LYNGO: also check parent overlap
print(f"{'Year':<6} {'N_hyb':<8} {'Novel%':<10} {'Novel_w/known_P1%':<20} {'Novel_w/both_P%':<20} {'N_envs':<8}")

for yr in range(2016, 2025):
    if yr == 2024:
        val_data = X_test
    else:
        val_data = X_train[X_train["year"] == yr]
    
    train_before = X_train[X_train["year"] < min(yr, 2024)]
    if yr > 2014:
        prior_h = set(train_before["hybrid"].unique())
        prior_parents = set(train_before["parent1"].dropna().unique()) | set(train_before["parent2"].dropna().unique())
    else:
        prior_h = set()
        prior_parents = set()
    
    val_h = set(val_data["hybrid"].unique())
    novel_h = val_h - prior_h
    
    # Among novel hybrids, parent coverage
    if novel_h:
        novel_mask = val_data["hybrid"].isin(novel_h)
        novel_data = val_data[novel_mask]
        known_p1 = novel_data["parent1"].isin(prior_parents).mean() * 100
        both_p = ((novel_data["parent1"].isin(prior_parents)) & 
                  (novel_data["parent2"].isin(prior_parents))).mean() * 100
    else:
        known_p1 = 0
        both_p = 0
    
    print(f"{yr:<6} {len(val_h):<8} {100*len(novel_h)/max(len(val_h),1):<10.1f} "
          f"{known_p1:<20.1f} {both_p:<20.1f} {val_data['Env'].nunique():<8}")

print()
print("=" * 70)
print("SUMMARY & RECOMMENDATION")
print("=" * 70)
print("""
ROOT CAUSE: The 2024 test combines TWO distribution shifts simultaneously:
  1. ~90% novel hybrids (never seen in training)
  2. Novel year (2024 weather conditions at all locations)

WHY EXISTING SCHEMES FAIL:
  - LEO: Tests spatial generalization but with KNOWN hybrids → misses the genetic novelty dimension
  - LGO: Tests genetic generalization but with KNOWN environments → misses the temporal dimension
  - LYO: Tests temporal generalization but the historical years have LOW novel-hybrid rates
         compared to the 2024 test

PROPOSED: Leave-Year-and-Novel-Genotype-Out (LYNGO)
  For validation year Y:
    1. Train on years < Y
    2. From year Y, identify hybrids NOT seen in years < Y
    3. Validate on ONLY those novel-hybrid rows in year Y
    4. Compute env_avg_pearson on this subset

  This replicates the actual test challenge: predicting novel hybrids in a novel year.
  Multiple holdout years (e.g., 2019-2023) give 5 folds, like rolling CV.

  KEY ADVANTAGE: Each fold faithfully simulates the SAME TYPE of prediction
  the model must make on the 2024 test — novel genotypes in a novel year.
""")
