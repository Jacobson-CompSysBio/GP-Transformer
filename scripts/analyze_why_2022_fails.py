#!/usr/bin/env python3
"""
Investigate WHY the 2022 holdout (92% novel hybrids) fails to track
the 2024 test (90% novel hybrids) despite similar novelty rates.

Goal: identify structural differences that make 2022 a poor proxy for 2024,
and explore whether ANY single-year holdout can be a reliable proxy.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr, spearmanr

DATA = Path("data/maize_data_2014-2023_vs_2024_v2")


def load():
    X_train = pd.read_csv(DATA / "X_train.csv")
    X_test = pd.read_csv(DATA / "X_test.csv")
    y_train = pd.read_csv(DATA / "y_train.csv")
    y_test = pd.read_csv(DATA / "y_test.csv")

    # Parse Env -> year, location, hybrid
    for df in [X_train, X_test]:
        df["Year"] = df["Env"].str.extract(r"_(\d{4})$").astype(int)
        # Location = everything before last _ (year)
        df["Location"] = df["Env"].str.replace(r"_\d{4}$", "", regex=True)
        # Hybrid from id: DEH1_2014-M0088/LH185 -> M0088/LH185
        df["Hybrid"] = df["id"].str.split("-", n=1).str[1]

    y_train["Yield"] = pd.to_numeric(y_train["Yield_Mg_ha"], errors="coerce")
    y_test["Yield"] = pd.to_numeric(y_test["Yield_Mg_ha"], errors="coerce")

    return X_train, X_test, y_train, y_test


def extract_parents(hybrid_name):
    """Split hybrid into parent1/parent2. Convention: '/' separator."""
    if "/" in hybrid_name:
        parts = hybrid_name.split("/", 1)
        return parts[0].strip(), parts[1].strip()
    return hybrid_name, None


def section(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def main():
    X_train, X_test, y_train, y_test = load()
    print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

    # Merge yield
    X_train = X_train.merge(y_train[["id", "Yield"]], on="id", how="left")
    X_test = X_test.merge(y_test[["id", "Yield"]], on="id", how="left")

    snp_cols = [c for c in X_train.columns if c.startswith("S")]
    ec_cols = [c for c in X_train.columns if not c.startswith("S") and c not in
               ["id", "Env", "Year", "Location", "Hybrid", "Yield"]]

    print(f"SNP columns: {len(snp_cols)}, EC/other columns: {len(ec_cols)}")

    # Get per-year data
    years = sorted(X_train["Year"].unique())
    hybrids_by_year = {y: set(X_train.loc[X_train.Year == y, "Hybrid"]) for y in years}
    hybrids_test = set(X_test["Hybrid"])
    all_train_hybrids = set(X_train["Hybrid"])

    # ─────────────────────────────────────────────────────────────────────
    section("1. TRAINING DATA AVAILABLE: 2022-holdout vs 2024-test")
    # ─────────────────────────────────────────────────────────────────────
    # When holding out 2022: train on 2014-2021 (8 years)
    # When predicting 2024: train on 2014-2023 (10 years)
    train_for_2022 = X_train[X_train.Year <= 2021]
    train_for_2024 = X_train  # all of 2014-2023

    print(f"Training for 2022 holdout: {len(train_for_2022)} rows, years 2014-2021")
    print(f"Training for 2024 test:    {len(train_for_2024)} rows, years 2014-2023")
    print(f"Difference: {len(train_for_2024) - len(train_for_2022)} extra rows ({(len(train_for_2024) - len(train_for_2022))/len(train_for_2022)*100:.1f}% more)")
    print()
    hybrids_train_2022 = set(train_for_2022["Hybrid"])
    hybrids_train_2024 = set(train_for_2024["Hybrid"])
    print(f"Unique hybrids for 2022 training: {len(hybrids_train_2022)}")
    print(f"Unique hybrids for 2024 training: {len(hybrids_train_2024)}")
    print(f"Extra hybrids in 2024 training: {len(hybrids_train_2024 - hybrids_train_2022)}")

    # ─────────────────────────────────────────────────────────────────────
    section("2. PARENTAL STRUCTURE: 2022-novel vs 2024-novel")
    # ─────────────────────────────────────────────────────────────────────
    val_2022 = X_train[X_train.Year == 2022]
    val_2022_hybrids = set(val_2022["Hybrid"])
    novel_2022 = val_2022_hybrids - hybrids_train_2022
    novel_2024 = hybrids_test - all_train_hybrids

    print(f"\n2022 holdout: {len(val_2022_hybrids)} hybrids, {len(novel_2022)} novel ({len(novel_2022)/len(val_2022_hybrids)*100:.1f}%)")
    print(f"2024 test:    {len(hybrids_test)} hybrids, {len(novel_2024)} novel ({len(novel_2024)/len(hybrids_test)*100:.1f}%)")

    # Parse parents
    parents_novel_2022 = {h: extract_parents(h) for h in novel_2022}
    parents_novel_2024 = {h: extract_parents(h) for h in novel_2024}

    p1_2022 = Counter(p1 for p1, _ in parents_novel_2022.values())
    p2_2022 = Counter(p2 for _, p2 in parents_novel_2022.values() if p2)
    p1_2024 = Counter(p1 for p1, _ in parents_novel_2024.values())
    p2_2024 = Counter(p2 for _, p2 in parents_novel_2024.values() if p2)

    print(f"\n2022 novel parent1 lines: {len(p1_2022)} unique")
    print(f"2024 novel parent1 lines: {len(p1_2024)} unique")
    p1_shared = set(p1_2022.keys()) & set(p1_2024.keys())
    print(f"Shared parent1 lines: {len(p1_shared)}")
    
    print(f"\n2022 novel parent2 (tester) lines: {len(p2_2022)} unique")
    print(f"2024 novel parent2 (tester) lines: {len(p2_2024)} unique")
    p2_shared = set(p2_2022.keys()) & set(p2_2024.keys())
    print(f"Shared parent2 lines: {len(p2_shared)}")

    print("\n2022 novel top testers (parent2):")
    for p, c in p2_2022.most_common(5):
        print(f"  {p}: {c} hybrids")
    print("2024 novel top testers (parent2):")
    for p, c in p2_2024.most_common(5):
        print(f"  {p}: {c} hybrids")

    # Parent overlap in training
    train_parents_2022 = set()
    for h in hybrids_train_2022:
        p1, p2 = extract_parents(h)
        train_parents_2022.add(p1)
        if p2:
            train_parents_2022.add(p2)

    train_parents_2024 = set()
    for h in hybrids_train_2024:
        p1, p2 = extract_parents(h)
        train_parents_2024.add(p1)
        if p2:
            train_parents_2024.add(p2)

    novel_p1_in_train_2022 = sum(1 for h in novel_2022 for p1, _ in [extract_parents(h)] if p1 in train_parents_2022)
    novel_p1_in_train_2024 = sum(1 for h in novel_2024 for p1, _ in [extract_parents(h)] if p1 in train_parents_2024)

    novel_any_p_in_train_2022 = sum(
        1 for h in novel_2022 
        for p1, p2 in [extract_parents(h)]
        if p1 in train_parents_2022 or (p2 and p2 in train_parents_2022)
    )
    novel_any_p_in_train_2024 = sum(
        1 for h in novel_2024
        for p1, p2 in [extract_parents(h)]
        if p1 in train_parents_2024 or (p2 and p2 in train_parents_2024)
    )

    print(f"\n2022 novel hybrids with parent1 in training: {novel_p1_in_train_2022}/{len(novel_2022)} ({novel_p1_in_train_2022/max(1,len(novel_2022))*100:.1f}%)")
    print(f"2024 novel hybrids with parent1 in training: {novel_p1_in_train_2024}/{len(novel_2024)} ({novel_p1_in_train_2024/max(1,len(novel_2024))*100:.1f}%)")
    print(f"\n2022 novel hybrids with ANY parent in training: {novel_any_p_in_train_2022}/{len(novel_2022)} ({novel_any_p_in_train_2022/max(1,len(novel_2022))*100:.1f}%)")
    print(f"2024 novel hybrids with ANY parent in training: {novel_any_p_in_train_2024}/{len(novel_2024)} ({novel_any_p_in_train_2024/max(1,len(novel_2024))*100:.1f}%)")

    # ─────────────────────────────────────────────────────────────────────
    section("3. GENETIC DISTANCE: SNP profiles of novel hybrids vs training")
    # ─────────────────────────────────────────────────────────────────────
    # Mean SNP profile per hybrid, then compare novel vs training centroids
    # Use a random sample for speed
    np.random.seed(42)

    def mean_snp_profile(df, hyb_set, snp_cols, max_hybrids=200):
        sub = df[df.Hybrid.isin(hyb_set)]
        hybs = list(sub.Hybrid.unique())
        if len(hybs) > max_hybrids:
            hybs = list(np.random.choice(hybs, max_hybrids, replace=False))
        profiles = []
        for h in hybs:
            row = sub[sub.Hybrid == h][snp_cols].mean(axis=0).values
            profiles.append(row)
        return np.array(profiles) if profiles else np.zeros((0, len(snp_cols)))

    # Compare centroid distance
    train_snps = X_train[snp_cols].mean(axis=0).values
    novel_2022_snps = val_2022[val_2022.Hybrid.isin(novel_2022)][snp_cols].mean(axis=0).values
    novel_2024_snps = X_test[X_test.Hybrid.isin(novel_2024)][snp_cols].mean(axis=0).values

    dist_train_2022 = cosine_dist(train_snps, novel_2022_snps)
    dist_train_2024 = cosine_dist(train_snps, novel_2024_snps)
    dist_2022_2024 = cosine_dist(novel_2022_snps, novel_2024_snps)

    print(f"Cosine distance (centroid of allele freqs):")
    print(f"  All training ↔ Novel 2022: {dist_train_2022:.6f}")
    print(f"  All training ↔ Novel 2024: {dist_train_2024:.6f}")
    print(f"  Novel 2022   ↔ Novel 2024: {dist_2022_2024:.6f}")

    # Per-SNP allele frequency comparison
    train_af = X_train[snp_cols].mean(axis=0)
    novel_2022_af = val_2022[val_2022.Hybrid.isin(novel_2022)][snp_cols].mean(axis=0)
    novel_2024_af = X_test[X_test.Hybrid.isin(novel_2024)][snp_cols].mean(axis=0)

    af_corr_train_2022 = pearsonr(train_af, novel_2022_af)[0]
    af_corr_train_2024 = pearsonr(train_af, novel_2024_af)[0]
    af_corr_2022_2024 = pearsonr(novel_2022_af, novel_2024_af)[0]

    print(f"\nAllele frequency correlations:")
    print(f"  All training ↔ Novel 2022: r={af_corr_train_2022:.4f}")
    print(f"  All training ↔ Novel 2024: r={af_corr_train_2024:.4f}")
    print(f"  Novel 2022   ↔ Novel 2024: r={af_corr_2022_2024:.4f}")

    # MAF shift — how many SNPs have large AF changes
    af_shift_2022 = (novel_2022_af - train_af).abs()
    af_shift_2024 = (novel_2024_af - train_af).abs()
    print(f"\nSNPs with |AF shift| > 0.1 from training:")
    print(f"  Novel 2022: {(af_shift_2022 > 0.1).sum()} / {len(snp_cols)}")
    print(f"  Novel 2024: {(af_shift_2024 > 0.1).sum()} / {len(snp_cols)}")
    print(f"SNPs with |AF shift| > 0.2:")
    print(f"  Novel 2022: {(af_shift_2022 > 0.2).sum()} / {len(snp_cols)}")
    print(f"  Novel 2024: {(af_shift_2024 > 0.2).sum()} / {len(snp_cols)}")

    # ─────────────────────────────────────────────────────────────────────
    section("4. ENVIRONMENTAL CONDITIONS: 2022 vs 2024 weather/EC features")
    # ─────────────────────────────────────────────────────────────────────
    if ec_cols:
        # Per-location comparison of EC features
        shared_locs = set(val_2022.Location) & set(X_test.Location)
        print(f"Shared locations between 2022 and 2024: {len(shared_locs)}")

        ec_diffs = []
        for loc in sorted(shared_locs):
            ec_2022 = val_2022[val_2022.Location == loc][ec_cols].apply(pd.to_numeric, errors='coerce').mean(axis=0)
            ec_2024 = X_test[X_test.Location == loc][ec_cols].apply(pd.to_numeric, errors='coerce').mean(axis=0)
            diff = (ec_2022 - ec_2024).abs().mean()
            ec_diffs.append(diff)
            
        print(f"Mean |EC feature diff| across shared locations: {np.mean(ec_diffs):.4f}")
        print(f"Std: {np.std(ec_diffs):.4f}, Max: {np.max(ec_diffs):.4f}")

        # Compare to year-over-year EC variability
        print("\nYear-over-year EC feature variability (at shared locations):")
        for y1 in range(2018, 2024):
            y2 = y1 + 1
            locs_y1 = set(X_train[X_train.Year == y1].Location)
            locs_y2_df = X_train[X_train.Year == y2] if y2 <= 2023 else X_test
            locs_y2 = set(locs_y2_df.Location)
            shared = locs_y1 & locs_y2
            if not shared:
                continue
            diffs = []
            for loc in shared:
                ec_y1 = X_train[(X_train.Year == y1) & (X_train.Location == loc)][ec_cols].apply(pd.to_numeric, errors='coerce').mean(axis=0)
                ec_y2_df = X_train if y2 <= 2023 else X_test
                ec_y2 = ec_y2_df[(ec_y2_df.Year == y2) & (ec_y2_df.Location == loc)][ec_cols].apply(pd.to_numeric, errors='coerce').mean(axis=0)
                diffs.append((ec_y1 - ec_y2).abs().mean())
            print(f"  {y1}→{y2}: mean |diff|={np.mean(diffs):.4f} (n_locs={len(shared)})")
    else:
        print("No EC columns found — skipping environmental comparison")

    # ─────────────────────────────────────────────────────────────────────
    section("5. YIELD DISTRIBUTION: 2022 vs 2024")
    # ─────────────────────────────────────────────────────────────────────
    yield_2022 = val_2022["Yield"].dropna()
    yield_2024 = X_test["Yield"].dropna()

    print(f"2022: mean={yield_2022.mean():.3f}, std={yield_2022.std():.3f}, n={len(yield_2022)}")
    print(f"2024: mean={yield_2024.mean():.3f}, std={yield_2024.std():.3f}, n={len(yield_2024)}")

    # Per-environment yield stats
    print("\n2022 per-env yield (novel hybrids only):")
    novel_2022_df = val_2022[val_2022.Hybrid.isin(novel_2022)]
    for env, grp in novel_2022_df.groupby("Env"):
        y = grp["Yield"].dropna()
        if len(y) >= 10:
            print(f"  {env}: mean={y.mean():.2f}, std={y.std():.2f}, n={len(y)}")
    
    print("\n2024 per-env yield (novel hybrids only):")
    novel_2024_df = X_test[X_test.Hybrid.isin(novel_2024)]
    for env, grp in novel_2024_df.groupby("Env"):
        y = grp["Yield"].dropna()
        if len(y) >= 10:
            print(f"  {env}: mean={y.mean():.2f}, std={y.std():.2f}, n={len(y)}")

    # ─────────────────────────────────────────────────────────────────────
    section("6. THE REAL PROBLEM: BREEDING PROGRAM STRUCTURE")
    # ─────────────────────────────────────────────────────────────────────
    # Check if 2022 and 2024 novel hybrids come from the same or different
    # breeding programs (detectable by tester lines)
    print("Tester (parent2) distribution for novel hybrids by year:")
    for yr in [2016, 2018, 2020, 2022]:
        yr_hybrids = hybrids_by_year[yr]
        cum_hybrids = set()
        for y in range(2014, yr):
            cum_hybrids |= hybrids_by_year[y]
        novel = yr_hybrids - cum_hybrids
        testers = Counter()
        for h in novel:
            _, p2 = extract_parents(h)
            if p2:
                testers[p2] += 1
        print(f"\n  {yr} ({len(novel)} novel hybrids):")
        for p, c in testers.most_common(5):
            print(f"    {p}: {c} ({c/max(len(novel),1)*100:.1f}%)")

    print(f"\n  2024 ({len(novel_2024)} novel hybrids):")
    for p, c in p2_2024.most_common(5):
        print(f"    {p}: {c} ({c/max(len(novel_2024),1)*100:.1f}%)")

    # ─────────────────────────────────────────────────────────────────────
    section("7. CROSS-YEAR FOLD SIMILARITY TO 2024 TEST")
    # ─────────────────────────────────────────────────────────────────────
    # For each possible holdout year, compute multiple similarity metrics
    # to the 2024 test setting
    print(f"{'Year':>4}  {'Novel%':>7}  {'N_novel':>7}  {'Tester_overlap':>15}  "
          f"{'AF_corr':>8}  {'AF_shift>0.1':>13}  {'P1_overlap':>11}  {'Score':>6}")
    print("-" * 90)

    test_testers = set(p2_2024.keys())

    for yr in range(2016, 2024):
        cum_hybrids = set()
        for y in range(2014, yr):
            cum_hybrids |= hybrids_by_year[y]
        yr_hybrids = hybrids_by_year[yr]
        novel = yr_hybrids - cum_hybrids
        
        if len(novel) < 5:
            continue

        # Tester overlap with 2024 test
        yr_testers = Counter()
        yr_p1 = Counter()
        for h in novel:
            p1, p2 = extract_parents(h)
            yr_p1[p1] += 1
            if p2:
                yr_testers[p2] += 1
        
        tester_overlap = len(set(yr_testers.keys()) & test_testers)
        total_testers_test = len(test_testers)
        
        # P1 overlap  
        p1_overlap = len(set(yr_p1.keys()) & set(p1_2024.keys()))

        # Allele frequency correlation
        yr_novel_df = X_train[(X_train.Year == yr) & (X_train.Hybrid.isin(novel))]
        if len(yr_novel_df) > 0:
            yr_af = yr_novel_df[snp_cols].mean(axis=0)
            af_corr = pearsonr(yr_af, novel_2024_af)[0]
            af_shift = (yr_af - novel_2024_af).abs()
            n_shift = (af_shift > 0.1).sum()
        else:
            af_corr = float("nan")
            n_shift = 0

        novel_rate = len(novel) / len(yr_hybrids) * 100

        # Composite similarity score (simple weighted)
        score = (
            0.3 * min(novel_rate / 90.0, 1.0) +           # novelty rate match
            0.2 * (tester_overlap / max(total_testers_test, 1)) +  # tester coverage
            0.2 * max(0, af_corr) +                        # genetic similarity
            0.15 * (p1_overlap / max(len(p1_2024), 1)) +   # parent1 coverage
            0.15 * max(0, 1 - n_shift / len(snp_cols))     # AF stability
        )

        print(f"{yr:4d}  {novel_rate:6.1f}%  {len(novel):7d}  "
              f"{tester_overlap}/{total_testers_test:>3d}          "
              f"{af_corr:8.4f}  {n_shift:>5d}/{len(snp_cols):<5d}  "
              f"{p1_overlap:>5d}/{len(p1_2024):<4d}  {score:6.3f}")

    # Also compute for 2024 test itself (as reference)
    print(f"\n2024 test reference: {len(novel_2024)} novel / {len(hybrids_test)} total "
          f"= {len(novel_2024)/len(hybrids_test)*100:.1f}%")

    # ─────────────────────────────────────────────────────────────────────
    section("8. FUNDAMENTAL ISSUE: N=1 TEST SET")
    # ─────────────────────────────────────────────────────────────────────
    print("""
The core problem is that we have ONE test set (2024) and we're trying to find
a validation set that "tracks" it. Even LYNGO is fundamentally limited because:

1. Each holdout year has a DIFFERENT breeding panel with DIFFERENT genetic
   architecture. 2022 novel hybrids ≠ 2024 novel hybrids in genetic makeup.

2. Each year has DIFFERENT weather. The GxE interaction that matters in 2022
   is different from the GxE interaction in 2024.

3. The training data CHANGES between folds. Training on 2014-2021 to predict
   2022 is a fundamentally different task than training on 2014-2023 to 
   predict 2024.

4. "Tracking" requires RANK PRESERVATION across configs. Even if two folds
   simulate similar challenges, the RELATIVE difficulty for different model
   configs may differ (e.g., MoE may help more in some genetic backgrounds
   than others).

This means: no single holdout year is guaranteed to produce the same
hyperparameter ranking as the 2024 test, regardless of novelty rate matching.
""")

    # ─────────────────────────────────────────────────────────────────────
    section("9. ALTERNATIVE: WHAT IF VALIDATION ISN'T FOR MODEL SELECTION?")
    # ─────────────────────────────────────────────────────────────────────
    print("""
If no validation scheme reliably RANKS models, consider alternatives:

A) ENSEMBLING over reasonable configs
   - Don't pick ONE best config; average predictions from 3-5 diverse configs
   - Each config may overfit differently; averaging cancels out idiosyncratic errors
   - Cost: 3-5× a single full run (same as LYNGO but produces predictions directly)

B) USE VALIDATION FOR EARLY STOPPING ONLY, NOT MODEL SELECTION
   - Fix architecture/HP from prior knowledge or literature
   - Use rolling/LEO val ONLY to decide when to stop training
   - This requires much less "tracking" — just needs the val curve to plateau
     near the right epoch, not rank configs correctly

C) COMPUTE PER-FOLD TEST EVAL IN ROLLING CV
   - In the existing rolling CV (100 epochs, 5 folds), each fold also evaluated
     on the 2024 test. But 100 epochs is too short.
   - If we ran 1-2 configs at FULL budget with rolling CV, we could check if
     any fold's val metric correlates with test at convergence.
   - Cost: 2 folds × 2 configs × 3000 epochs = 4 full runs

D) FOCUS ON ROBUST APPROACHES RATHER THAN TUNING
   - GBLUP / ridge-regression baselines perform consistently
   - Transformer advantages come from modelling GxE, which is EXACTLY what's
     hard to validate when the environment is novel
   - Maybe the right question is: "what's the simplest model that captures
     parent-level effects reliably?" rather than "which config is best?"
""")

    # ─────────────────────────────────────────────────────────────────────
    section("10. QUANTITATIVE CHECK: HOW DIFFERENT IS EACH YEAR'S 'PREDICTION PROBLEM'?")
    # ─────────────────────────────────────────────────────────────────────
    # For each year Y with high novelty:
    #   - What fraction of novel hybrids have half-sibs in training?
    #   - What's the genetic diversity (mean pairwise distance) of the novel panel?
    #   - How many environments are shared with the test?
    
    print(f"\n{'Year':>4}  {'N_novel':>7}  {'HalfSib%':>9}  {'MeanYield':>10}  {'StdYield':>9}  "
          f"{'N_envs':>6}  {'Shared_w_2024':>13}")
    print("-" * 75)

    test_locs = set(X_test.Location)
    
    for yr in [2016, 2018, 2020, 2022, 2024]:
        cum_hybrids = set()
        cum_parents = set()
        for y in range(2014, yr):
            cum_hybrids |= hybrids_by_year.get(y, set())
            for h in hybrids_by_year.get(y, set()):
                p1, p2 = extract_parents(h)
                cum_parents.add(p1)
                if p2:
                    cum_parents.add(p2)
        
        if yr == 2024:
            yr_df = X_test
            yr_hybs = hybrids_test
        else:
            yr_df = X_train[X_train.Year == yr]
            yr_hybs = hybrids_by_year[yr]
        
        novel = yr_hybs - cum_hybrids
        
        # Half-sib rate: novel hybrid has at least one parent that appears
        # as a parent in DIFFERENT hybrids in training
        half_sib_count = 0
        for h in novel:
            p1, p2 = extract_parents(h)
            if p1 in cum_parents or (p2 and p2 in cum_parents):
                half_sib_count += 1
        
        novel_df = yr_df[yr_df.Hybrid.isin(novel)]
        y_vals = novel_df["Yield"].dropna()
        yr_locs = set(yr_df.Location)
        shared_locs = yr_locs & test_locs
        
        print(f"{yr:4d}  {len(novel):7d}  {half_sib_count/max(len(novel),1)*100:8.1f}%  "
              f"{y_vals.mean():10.3f}  {y_vals.std():9.3f}  "
              f"{len(yr_locs):6d}  {len(shared_locs):>5d}/{len(test_locs)}")

    # ─────────────────────────────────────────────────────────────────────
    section("11. HALF-SIB NETWORK CONNECTIVITY")
    # ─────────────────────────────────────────────────────────────────────
    # For each holdout year's novel hybrids, count: how many half-sibs
    # (hybrids sharing one parent) exist in the training data?
    # More half-sib observations = more information to predict from
    
    print("For each holdout year's novel hybrids, how many training ROWS")
    print("share at least one parent (half-sib connectivity)?\n")

    for yr in [2016, 2018, 2020, 2022]:
        cum_hybrids = set()
        for y in range(2014, yr):
            cum_hybrids |= hybrids_by_year[y]
        yr_hybs = hybrids_by_year[yr]
        novel = yr_hybs - cum_hybrids
        
        train_df = X_train[X_train.Year < yr]
        
        # Collect parent→rows mapping in training
        parent_rows = {}
        for _, row in train_df[["Hybrid"]].drop_duplicates().iterrows():
            h = row["Hybrid"]
            p1, p2 = extract_parents(h)
            parent_rows.setdefault(p1, set()).add(h)
            if p2:
                parent_rows.setdefault(p2, set()).add(h)
        
        total_half_sibs = 0
        connected = 0
        for h in novel:
            p1, p2 = extract_parents(h)
            sibs = set()
            if p1 in parent_rows:
                sibs |= parent_rows[p1]
            if p2 and p2 in parent_rows:
                sibs |= parent_rows[p2]
            sibs.discard(h)
            if sibs:
                connected += 1
                total_half_sibs += len(sibs)
        
        avg_sibs = total_half_sibs / max(connected, 1)
        print(f"  Holdout {yr}: {connected}/{len(novel)} novel hybrids connected "
              f"({connected/max(len(novel),1)*100:.1f}%), avg {avg_sibs:.1f} half-sib hybrids in training")
    
    # Same for 2024
    train_df = X_train
    parent_rows = {}
    for _, row in train_df[["Hybrid"]].drop_duplicates().iterrows():
        h = row["Hybrid"]
        p1, p2 = extract_parents(h)
        parent_rows.setdefault(p1, set()).add(h)
        if p2:
            parent_rows.setdefault(p2, set()).add(h)
    
    total_half_sibs = 0
    connected = 0
    for h in novel_2024:
        p1, p2 = extract_parents(h)
        sibs = set()
        if p1 in parent_rows:
            sibs |= parent_rows[p1]
        if p2 and p2 in parent_rows:
            sibs |= parent_rows[p2]
        sibs.discard(h)
        if sibs:
            connected += 1
            total_half_sibs += len(sibs)
    
    avg_sibs = total_half_sibs / max(connected, 1)
    print(f"  Test 2024:   {connected}/{len(novel_2024)} novel hybrids connected "
          f"({connected/max(len(novel_2024),1)*100:.1f}%), avg {avg_sibs:.1f} half-sib hybrids in training")


if __name__ == "__main__":
    main()
