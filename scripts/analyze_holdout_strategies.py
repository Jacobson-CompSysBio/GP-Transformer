#!/usr/bin/env python3
"""
Address two counterarguments:
1. If 2022 is HARDER (fewer half-sibs), do models that do well on 2022 do poorly on 2024?
   → Check if there's an INVERSE correlation between difficulty and performance.
2. If 2024 is "just interpolation from rich half-sibs," then 2023 (which reuses 2022 panel)
   should work as validation. But it doesn't. Why?

Also: propose and evaluate novel holdout strategies.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.stats import pearsonr

DATA = Path("data/maize_data_2014-2023_vs_2024_v2")


def load():
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_test = pd.read_csv(DATA / "X_test.csv")
    y_train = pd.read_csv(DATA / "y_train.csv")
    y_test = pd.read_csv(DATA / "y_test.csv")
    for df in [X_train, X_test]:
        df["Year"] = df["Env"].str.extract(r"_(\d{4})$").astype(int)
        df["Location"] = df["Env"].str.replace(r"_\d{4}$", "", regex=True)
        df["Hybrid"] = df["id"].str.split("-", n=1).str[1]
    y_train["Yield"] = pd.to_numeric(y_train["Yield_Mg_ha"], errors="coerce")
    y_test["Yield"] = pd.to_numeric(y_test["Yield_Mg_ha"], errors="coerce")
    return X_train, X_test, y_train, y_test


def extract_parents(h):
    if "/" in str(h):
        parts = str(h).split("/", 1)
        return parts[0].strip(), parts[1].strip()
    return str(h), None


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    X_train, X_test, y_train, y_test = load()
    X_train = X_train.merge(y_train[["id", "Yield"]], on="id", how="left")
    X_test = X_test.merge(y_test[["id", "Yield"]], on="id", how="left")

    snp_cols = [c for c in X_train.columns if c.startswith("S")]
    years = sorted(X_train["Year"].unique())
    hybrids_by_year = {y: set(X_train.loc[X_train.Year == y, "Hybrid"]) for y in years}
    hybrids_test = set(X_test["Hybrid"])
    all_train_hybrids = set(X_train["Hybrid"])

    # ─────────────────────────────────────────────────────────────────
    section("1. IS 2022 A HARDER PREDICTION PROBLEM THAN 2024?")
    # ─────────────────────────────────────────────────────────────────
    print("""
If 2022 is genuinely harder (7.2 vs 186.0 avg half-sibs), we'd expect:
- Lower absolute prediction accuracy on 2022
- The SAME model performing worse on 2022 than 2024
- But question: does "harder" for ALL models mean the RANKING changes?

Key distinction:
- If 2022 is uniformly harder for ALL configs → rankings preserved, validation works
- If 2022 is differentially harder (some configs collapse, others don't) → rankings break
""")

    # Characterize difficulty dimensions
    for yr, label in [(2022, "HOLDOUT 2022"), (2024, "TEST 2024")]:
        cum_hybrids = set()
        cum_parents = set()
        for y in range(2014, yr):
            cum_hybrids |= hybrids_by_year.get(y, set())
            for h in hybrids_by_year.get(y, set()):
                p1, p2 = extract_parents(h)
                cum_parents.add(p1)
                if p2: cum_parents.add(p2)

        if yr == 2024:
            yr_hybs = hybrids_test
            yr_df = X_test
        else:
            yr_hybs = hybrids_by_year[yr]
            yr_df = X_train[X_train.Year == yr]

        novel = yr_hybs - cum_hybrids
        novel_df = yr_df[yr_df.Hybrid.isin(novel)]

        # Half-sib stats
        parent_rows = {}
        train_df = X_train[X_train.Year < yr] if yr <= 2023 else X_train
        for h in set(train_df.Hybrid):
            p1, p2 = extract_parents(h)
            parent_rows.setdefault(p1, set()).add(h)
            if p2: parent_rows.setdefault(p2, set()).add(h)

        sib_counts = []
        for h in novel:
            p1, p2 = extract_parents(h)
            sibs = set()
            if p1 in parent_rows: sibs |= parent_rows[p1]
            if p2 and p2 in parent_rows: sibs |= parent_rows[p2]
            sibs.discard(h)
            sib_counts.append(len(sibs))

        sib_arr = np.array(sib_counts)
        print(f"\n{label}:")
        print(f"  Novel hybrids: {len(novel)}")
        print(f"  Half-sibs per novel hybrid: mean={sib_arr.mean():.1f}, median={np.median(sib_arr):.1f}, "
              f"min={sib_arr.min()}, max={sib_arr.max()}")
        print(f"  Hybrids with 0 half-sibs: {(sib_arr == 0).sum()} ({(sib_arr == 0).mean()*100:.1f}%)")
        print(f"  Hybrids with <10 half-sibs: {(sib_arr < 10).sum()} ({(sib_arr < 10).mean()*100:.1f}%)")
        print(f"  Hybrids with >100 half-sibs: {(sib_arr > 100).sum()} ({(sib_arr > 100).mean()*100:.1f}%)")

        # Yield spread (proxy for signal/noise)
        y = novel_df["Yield"].dropna()
        print(f"  Yield: mean={y.mean():.2f}, std={y.std():.2f}, CV={y.std()/y.mean()*100:.1f}%")

        # Number of environments and per-env sample size
        env_sizes = novel_df.groupby("Env").size()
        print(f"  Environments: {len(env_sizes)}, median samples/env={env_sizes.median():.0f}")

    # ─────────────────────────────────────────────────────────────────
    section("2. THE 2023 COUNTERPOINT: WHY DOESN'T INTERPOLATION WORK?")
    # ─────────────────────────────────────────────────────────────────
    print("""
User's argument: If 2024 test is "just" interpolation from rich half-sib networks,
then 2023 holdout should work because:
- 2023 reuses 2022's panel (99.8% overlap) → same hybrids are "known"
- 2023 has novel weather → temporal generalization is tested
- When training on 2014-2022, 2023's hybrids are KNOWN (seen in 2022)
- But validation on 2023 STILL doesn't track 2024 test performance

This is a strong counterargument. Let's quantify the structural differences.
""")

    # 2023 holdout setup: train on 2014-2022, val on 2023
    hyb_2023 = hybrids_by_year[2023]
    hyb_train_for_2023 = set()
    for y in range(2014, 2023):
        hyb_train_for_2023 |= hybrids_by_year[y]

    novel_2023 = hyb_2023 - hyb_train_for_2023
    known_2023 = hyb_2023 & hyb_train_for_2023

    print(f"2023 holdout: {len(hyb_2023)} hybrids")
    print(f"  Novel: {len(novel_2023)} ({len(novel_2023)/len(hyb_2023)*100:.1f}%)")
    print(f"  Known: {len(known_2023)} ({len(known_2023)/len(hyb_2023)*100:.1f}%)")
    print()

    # The key: 2023 KNOWN hybrids → what's the model actually predicting?
    # It's predicting KNOWN hybrids in a novel environment (new weather)
    # But 2024 is predicting NOVEL hybrids in a novel environment
    print("2023 val tests: KNOWN hybrids × novel weather → environment extrapolation")
    print("2024 test tests: NOVEL hybrids × novel weather → genetic + environment extrapolation")
    print()

    # But wait — the user says LEO also tests known hybrids in novel envs, and it
    # doesn't track either. So the problem isn't just about known vs novel hybrids.

    # Let's check what LEO is ACTUALLY doing when it validates
    print("Key question: When LEO validates, what is it actually testing?")
    print()

    # LEO holds out entire environments. In the training data, these envs contain
    # hybrids that also appear in other environments.
    # So LEO tests: can the model predict yield of a KNOWN hybrid at a HELD-OUT location-year?
    # This is spatial interpolation + some temporal if multiple years are included.

    # 2023 holdout tests: can the model predict yield of a KNOWN hybrid in a novel year?
    # This is temporal extrapolation.

    # 2024 test tests: can the model predict yield of a NOVEL hybrid in a novel year?
    # This is genetic + temporal extrapolation.

    # None of them test the same thing!

    # ─── The deeper issue: what LATENT FACTOR determines if a config is good? ───
    print("The deeper issue: WHAT MODEL CAPABILITY is being tested?")
    print()
    print("  LEO val:    spatial interpolation with known genotypes")
    print("  2023 val:   temporal extrapolation with known genotypes")
    print("  2024 test:  temporal + genetic extrapolation with novel genotypes")
    print()
    print("  These require DIFFERENT model capabilities:")
    print("  - LEO: learning location effects + GxE interactions at known locs")
    print("  - 2023: learning year/weather effects while leveraging known hybrid embeddings")
    print("  - 2024: learning PARENTAL COMBINING ABILITY that transfers to novel crosses")
    print()

    # ─────────────────────────────────────────────────────────────────
    section("3. WHAT EXACTLY DOES 2024 PREDICTION REQUIRE?")
    # ─────────────────────────────────────────────────────────────────
    novel_2024 = hybrids_test - all_train_hybrids

    # For each novel 2024 hybrid, what information is available?
    # - SNP genotype (always available)
    # - Parent1 effects (if parent1 appeared in OTHER crosses in training)
    # - Parent2 effects (tester; always known since PHP02 and LH287 are common)
    # - Environmental covariates for 2024 locations (always available)

    # Count how many training OBSERVATIONS (not just hybrids) involve each tester
    tester_obs = {}
    for _, row in X_train[["Hybrid"]].drop_duplicates().iterrows():
        h = row["Hybrid"]
        _, p2 = extract_parents(h)
        if p2:
            tester_obs[p2] = tester_obs.get(p2, 0) + 1

    test_testers = Counter()
    for h in novel_2024:
        _, p2 = extract_parents(h)
        if p2:
            test_testers[p2] += 1

    print("2024 test relies on tester (parent2) combining ability:")
    for tester, count in test_testers.most_common():
        train_count = tester_obs.get(tester, 0)
        print(f"  {tester}: {count} novel hybrids in test, "
              f"{train_count} hybrids with this tester in training")

    # Count training ROWS (not unique hybrids) per tester
    print("\nTraining ROWS per tester:")
    for tester in test_testers.keys():
        rows = X_train[X_train.Hybrid.apply(lambda h: extract_parents(h)[1] == tester)]
        print(f"  {tester}: {len(rows)} rows across {rows.Year.nunique()} years, "
              f"{rows.Env.nunique()} environments")

    # ─────────────────────────────────────────────────────────────────
    section("4. NOVEL HOLDOUT STRATEGY IDEAS")
    # ─────────────────────────────────────────────────────────────────

    # ── Strategy A: Tester-Matched Holdout ──
    print("STRATEGY A: TESTER-MATCHED HOLDOUT")
    print("-" * 50)
    print("""
  Idea: Hold out the most recent year(s) that use the SAME testers as the 2024 test.
  2024 uses PHP02 (47%) and LH287 (53%).
  Find which training years have hybrids with these testers.
""")
    for tester in ["PHP02", "LH287"]:
        print(f"  Tester {tester} in training years:")
        for yr in years:
            yr_df = X_train[X_train.Year == yr]
            tester_hybs = [h for h in set(yr_df.Hybrid)
                          if extract_parents(h)[1] == tester]
            if tester_hybs:
                print(f"    {yr}: {len(tester_hybs)} hybrids")

    # ── Strategy B: Leave-Tester-Out ──
    print("\nSTRATEGY B: LEAVE-TESTER-OUT")
    print("-" * 50)
    print("""
  Idea: Instead of holding out a year, hold out all hybrids with a specific tester
  line from training. Then validate on those hybrids.
  This tests: can the model predict combining ability for a tester it hasn't seen?
  
  Advantage: Tests genetic generalization at the TESTER level, which is closer
  to what 2024 requires (novel crosses with known testers... but seen from
  the other side — known crosses with held-out testers).
  
  Problem: 2024 testers ARE in training. This tests the wrong direction.
""")

    # ── Strategy C: Synthetic Novel Hybrids ──
    print("\nSTRATEGY C: LEAVE-PARENT1-GROUP-OUT (by year)")
    print("-" * 50)
    print("""
  Idea: For a holdout year Y with high novelty (2018, 2020, 2022):
    - Train on all years EXCEPT Y
    - Validate on Y
    - The novel hybrids in Y have novel parent1 lines
    - This tests: can the model predict novel crosses given known testers?
  
  Key difference from LYO: we're NOT just holding out a year for temporal
  generalization — we're holding out a year BECAUSE it introduces novel
  parent1 lines that don't appear elsewhere.
  
  But we need to check: do the novel parent1 lines in 2022 vs 2024 have
  similar properties?
""")

    # Check parent1 properties
    for yr in [2018, 2020, 2022]:
        cum = set()
        for y in range(2014, yr):
            cum |= hybrids_by_year[y]

        novel = hybrids_by_year[yr] - cum
        p1_lines = set()
        for h in novel:
            p1, _ = extract_parents(h)
            p1_lines.add(p1)

        # How many of these parent1 lines appear in training at all
        # (Maybe in earlier years as testers? Or in other crosses?)
        all_train_parents = set()
        train_sub = X_train[X_train.Year < yr]
        for h in set(train_sub.Hybrid):
            p1, p2 = extract_parents(h)
            all_train_parents.add(p1)
            if p2: all_train_parents.add(p2)

        known_p1 = sum(1 for p in p1_lines if p in all_train_parents)
        print(f"  {yr}: {len(novel)} novel hybrids, {len(p1_lines)} unique parent1 lines, "
              f"{known_p1} ({known_p1/len(p1_lines)*100:.1f}%) appear elsewhere in training")

    novel_2024_p1 = set()
    for h in novel_2024:
        p1, _ = extract_parents(h)
        novel_2024_p1.add(p1)
    known_p1_2024 = sum(1 for p in novel_2024_p1 if p in set(p1 for h in all_train_hybrids for p1, _ in [extract_parents(h)]))
    print(f"  2024: {len(novel_2024)} novel hybrids, {len(novel_2024_p1)} unique parent1 lines, "
          f"{known_p1_2024} ({known_p1_2024/len(novel_2024_p1)*100:.1f}%) appear elsewhere in training")

    # ── Strategy D: Cross-Tester Holdout ──
    print("\nSTRATEGY D: CROSS-TESTER HOLDOUT (novel)")
    print("-" * 50)
    print("""
  Idea: From training data, construct a holdout set of hybrids where:
    1. The TESTER is known (PHP02 or LH287 — same as 2024 test)
    2. The PARENT1 is NOT in ANY other cross with that same tester in training
    
  This exactly replicates the 2024 challenge:
    - Model must predict a novel parent1 × known tester cross
    - Tester combining ability can be learned from other crosses
    - Parent1 SNP effects must be inferred from genotype alone (or from
      that parent1 appearing with DIFFERENT testers)
    
  Construction: For each PHP02 cross in training, check if parent1 appears
  in any other PHP02 cross. If not, it's a "novel cross" candidate.
""")
    for tester in ["PHP02", "LH287"]:
        # Find all hybrids with this tester
        tester_hybrids = []
        for h in all_train_hybrids:
            p1, p2 = extract_parents(h)
            if p2 == tester:
                tester_hybrids.append((h, p1))

        # For each parent1, count how many crosses with this tester exist
        p1_counts = Counter(p1 for _, p1 in tester_hybrids)
        unique_p1 = [p1 for p1, c in p1_counts.items() if c == 1]
        multi_p1 = [p1 for p1, c in p1_counts.items() if c > 1]

        print(f"\n  Tester={tester}: {len(tester_hybrids)} hybrids in training")
        print(f"    Parent1 lines with exactly 1 cross: {len(unique_p1)} (potential holdout)")
        print(f"    Parent1 lines with >1 cross: {len(multi_p1)}")

        # How many rows would the holdout have?
        holdout_hybs = [h for h, p1 in tester_hybrids if p1 in unique_p1]
        holdout_rows = X_train[X_train.Hybrid.isin(holdout_hybs)]
        print(f"    Holdout size: {len(holdout_hybs)} hybrids, {len(holdout_rows)} rows, "
              f"{holdout_rows.Env.nunique()} environments")

        # How many of these parent1 lines appear in OTHER crosses (different tester)?
        p1_in_other_cross = 0
        other_parents = set()
        for h in all_train_hybrids:
            p1, p2 = extract_parents(h)
            if p2 != tester:
                other_parents.add(p1)
        p1_in_other_cross = sum(1 for p1 in unique_p1 if p1 in other_parents)
        print(f"    Of holdout parent1s, {p1_in_other_cross}/{len(unique_p1)} "
              f"({p1_in_other_cross/max(len(unique_p1),1)*100:.1f}%) appear in crosses with OTHER testers")

    # ── Strategy E: Environment-Weighted Hybrid Holdout ──
    print("\n\nSTRATEGY E: LEAVE-RECENT-NOVEL-HYBRIDS-OUT (within same years)")
    print("-" * 50)
    print("""
  Idea: Don't hold out a year at all. Instead:
    1. From the most recent years (2022-2023), identify hybrids whose
       PARENT1 line is the same as 2024 test parent1 lines
    2. If no direct match, identify hybrids with the same TESTER as 2024
       (PHP02, LH287) and hold out a random subset
    3. Validate on this subset WITHIN the training years' environments
    
  Advantage: Same weather/locations as training, tests genetic generalization
  Disadvantage: Doesn't test temporal extrapolation
  
  Could be combined with year holdout for a more complete picture.
""")

    # Find hybrids with PHP02/LH287 as tester in recent years
    for yr in [2020, 2021, 2022, 2023]:
        yr_df = X_train[X_train.Year == yr]
        for tester in ["PHP02", "LH287"]:
            tester_hybs = [h for h in set(yr_df.Hybrid) if extract_parents(h)[1] == tester]
            if tester_hybs:
                rows = yr_df[yr_df.Hybrid.isin(tester_hybs)]
                print(f"  {yr}, tester={tester}: {len(tester_hybs)} hybrids, {len(rows)} rows")

    # ── Strategy F: Tester-Matched Year Holdout ──
    print("\n\nSTRATEGY F: TESTER-MATCHED YEAR HOLDOUT")
    print("-" * 50)
    print("""
  Idea: The closest structural match to 2024 would be a holdout year where:
    1. High novelty rate (>80%)  
    2. Same or similar tester lines as 2024 (PHP02, LH287)
    3. Rich half-sib connectivity
    
  Let's check which years match on ALL THREE criteria:
""")

    test_testers_set = {"PHP02", "LH287"}
    print(f"{'Year':>4}  {'Novel%':>7}  {'Testers':>30}  {'Tester∩2024':>12}  "
          f"{'AvgHalfSibs':>11}  {'N_rows':>7}")
    print("-" * 85)

    for yr in range(2016, 2024):
        cum = set()
        cum_parents = set()
        for y in range(2014, yr):
            cum |= hybrids_by_year[y]
            for h in hybrids_by_year[y]:
                p1, p2 = extract_parents(h)
                cum_parents.add(p1)
                if p2: cum_parents.add(p2)

        novel = hybrids_by_year[yr] - cum
        if len(novel) < 5:
            continue

        # Tester distribution
        testers = Counter()
        for h in novel:
            _, p2 = extract_parents(h)
            if p2: testers[p2] += 1
        top_testers = [t for t, _ in testers.most_common(3)]
        tester_str = ", ".join(f"{t}({c})" for t, c in testers.most_common(3))
        tester_overlap = len(set(testers.keys()) & test_testers_set)

        # Half-sib connectivity
        parent_hybs = {}
        train_sub = X_train[X_train.Year < yr]
        for h in set(train_sub.Hybrid):
            p1, p2 = extract_parents(h)
            parent_hybs.setdefault(p1, set()).add(h)
            if p2: parent_hybs.setdefault(p2, set()).add(h)

        sib_counts = []
        for h in novel:
            p1, p2 = extract_parents(h)
            sibs = set()
            if p1 in parent_hybs: sibs |= parent_hybs[p1]
            if p2 and p2 in parent_hybs: sibs |= parent_hybs[p2]
            sibs.discard(h)
            sib_counts.append(len(sibs))
        avg_sibs = np.mean(sib_counts)

        novel_rate = len(novel) / len(hybrids_by_year[yr]) * 100
        yr_rows = X_train[(X_train.Year == yr) & (X_train.Hybrid.isin(novel))]

        print(f"{yr:4d}  {novel_rate:6.1f}%  {tester_str:>30s}  "
              f"{tester_overlap:>5d}/{len(test_testers_set)}       "
              f"{avg_sibs:10.1f}  {len(yr_rows):7d}")

    # ─────────────────────────────────────────────────────────────────
    section("5. ASSESSMENT: LIKELIHOOD OF TRACKING")
    # ─────────────────────────────────────────────────────────────────
    print("""
STRATEGY ASSESSMENT SUMMARY:

A) Tester-Matched Year Holdout
   Best candidate: 2020 (PHP02 is one of its 3 testers, 96% novel, avg 116 half-sibs)
   Likelihood of tracking: LOW-MODERATE
   Reason: 2020 uses PHK76/PHP02/PHZ51, not PHP02/LH287. Different tester
           mix means different GCA patterns being tested. Also 2020 has
           different weather. But it's the closest structural match.

B) Leave-Tester-Out
   Likelihood of tracking: LOW
   Reason: Tests the wrong direction of novelty (held-out tester vs held-out parent1)

C) Leave-Parent1-Group-Out  
   Likelihood of tracking: LOW-MODERATE
   Reason: Tests novel parent1 prediction, but within known environments.
           Missing the temporal extrapolation dimension.

D) Cross-Tester Holdout (MOST PROMISING)
   Likelihood of tracking: MODERATE
   Reason: Directly replicates the 2024 challenge — novel parent1 × known tester.
           Uses SAME testers (PHP02, LH287) from training.
   Caveat: Still within known environments/years, not testing weather generalization.
           But this is the ONLY strategy that tests the same GENETIC prediction task.

E) Leave-Recent-Novel-Hybrids-Out
   Likelihood of tracking: LOW
   Reason: Confounds temporal and genetic effects.

F) ENSEMBLE (not a holdout — sidesteps validation entirely)
   Likelihood of "tracking": N/A — produces predictions directly.
   Rationale: If we can't reliably SELECT configs, average over several.
   
BOTTOM LINE:
  No holdout set can simultaneously match 2024 on:
    (a) tester identity, (b) parent1 novelty, (c) half-sib connectivity,
    (d) weather conditions, AND (e) training data volume.
  
  Strategy D (Cross-Tester Holdout) matches (a), (b), (c) but not (d), (e).
  Strategy F (Ensemble) avoids the validation problem entirely.
""")


if __name__ == "__main__":
    main()
