# AGENTS.md — GP-Transformer development plan (code repo)

> **Scope.** This is the development plan of record for the **code** repo
> (`research/phenotyping/gp-transformer`). It is *not* the paper-repo reconciliation
> file (`latex/gxe-transformer-paper/AGENTS.md`), which governs the thesis chapter.
> This file supersedes the earlier scale-fix / attention-pool / uncertainty-weighting
> phased plan: the scale problem is now handled by the affine-calibration head, and
> mean-pooling has already been replaced by CLS pooling. The remaining objective is
> the headline metric.
>
> **Goal.** Surpass the competition-winning **0.45** macro env-PCC on the G2F 2024
> test set (currently 6th), and tighten absolute-scale RMSE as a secondary metric.
> **Baseline of record:** env-avg PCC **0.42321**, env-MSE **55.42457**, run
> `…_j4227430` on `origin/affine-shift`.
>
> **Canonical best config (do not lose this).** FullTransformer, unified sequence
> `CLS + 2024 markers + 702 env = 2727` tokens; 1 transformer block, 4 heads, 256d;
> MoE 8 experts / top-2 + shared, aux 0.01; env-contrastive `mode=e`, w 0.1, τ 0.5;
> affine calibration (`calibration_mode=env_affine`); `envpcc` loss; LEO val 15%;
> env-stratified batching `MIN_SAMPLES_PER_ENV=32`; GBS 8192 / microbatch 256 over
> 4×8 GPUs; AdamW lr 1e-4→1e-5 cosine (warmup ~5 ep, horizon ~400 ep), wd 1e-5,
> dropout 0.15, seed 1.

---

## 0. Ground truth the plan must respect

Two diagnostics from this repo condition everything below. Re-read
`scripts/analyze_holdout_strategies.py`, `scripts/analyze_val_failure.py`, and
`scripts/fit_decomposition.py` before proposing changes.

1. **The test is a novel-cross + novel-year problem.** 2024 is ~90% novel hybrids
   (novel parent1 × *known* tester, mostly PHP02 / LH287) in 100% novel environments.
   Clearing 0.45 requires (a) parental **combining ability** that transfers to unseen
   crosses and (b) weather generalization to a new year — not raw capacity. The fact
   that the best model is **one block deep** is the strongest evidence that we are
   input/objective-bound, not parameter-bound.

2. **No current validation scheme tracks 2024.** LEO measures spatial interpolation
   with *known* hybrids; a 2023 holdout measures temporal extrapolation with *known*
   hybrids. Neither measures novel-hybrid × novel-year. **Until this is fixed, a flat
   result is unmeasurable, not negative.** This is why Phase 0 gates the plan.

---

## 1. Operating principles — *what we do when a change doesn't work*

The honest answer to "should changes that don't work just be dropped?" is **no, not
by default.** The right action depends on *why* it failed and *what kind of change*
it was. Dropping everything that doesn't immediately move 0.42→0.45 would discard
correctness fixes, throw away informative null results, and overfit the roadmap to a
metric we currently can't even measure reliably.

### 1.1 Pre-register every experiment

Before running, write three things into the **Decision Log** (§4):

- **Metric & estimator** — which number, measured on which split, averaged over how
  many seeds/folds (never a single seed; see §1.4).
- **Success bar** — the delta that counts as a win, stated *before* seeing results.
  Default bar: **+0.005 env-PCC over baseline, exceeding the seed-ensemble's ±1σ
  band.** A change inside the noise band is not a win, no matter how principled.
- **Pre-committed action for each outcome** — what KEEP / ITERATE / PARK / CUT means
  for *this* change (definitions below). This prevents post-hoc rationalization
  against a noisy validation signal.

### 1.2 Classify the change — different classes have different bars

| Class | Examples in this plan | Bar to keep | Default if it fails the bar |
|---|---|---|---|
| **Correctness fix** | cross-GPU env-PCC stats; per-feature tokenizer (identity currently lives only in positional code) | *Principled correctness alone.* Keep even if headline is flat. | **KEEP** — do not revert to a known-worse implementation because a noisy metric didn't reward it. Log as variance-reduction. |
| **Additive-information feature** | temporal weather / envirotyping; parent embeddings; dual additive+dominance channel | Clears the success bar | **PARK behind a flag** + document the null. Do not delete. May only pay off once a later phase can use it (e.g. weather needs depth to exploit). |
| **Architecture-family swap** | Perceiver latent bottleneck; Mamba tracks; factorized cross-attention G×E | Clears the bar *and* the gain justifies added complexity/compute/maintenance | **CUT.** Complexity must earn its keep; unrewarded architecture is debt. |
| **Pretraining scheme** | marker MLM; cross-modal CLIP | Improves the *fine-tuned* metric, not just the pretext loss | **PARK** the recipe, keep the checkpoint if cheap to retain; revisit if the encoder is reused. |

### 1.3 The four decisions

- **KEEP** — meets its bar (or is a correctness fix). Merge behind a documented flag;
  set as a default only after it survives a re-run on a second seed set.
- **ITERATE** — partial signal, or a plausible implementation bug, or the success bar
  was nearly met. Allowed **at most two** iteration rounds before it must resolve to
  KEEP / PARK / CUT. This cap exists to stop infinite tuning against noise.
- **PARK** — properly measured, genuine null/regression, but the idea may interact
  with later phases. Keep the code behind a flag, write the null result in §4, move
  on. Parked items are re-evaluated once at the end of the phase that could unlock
  them.
- **CUT** — measured null/regression with no plausible future interaction, *or* a
  positive that isn't worth its complexity. Remove from the trunk; the Decision Log
  entry is the record.

### 1.4 Separate signal from noise *before* deciding

Because the validation signal is weak (§0.2), no decision is made on a single run.

- **Always seed-ensemble** (≥3 seeds) and report mean ± σ. The success bar is defined
  relative to this band.
- **Change one thing at a time.** Composite changes that move the metric must be
  decomposed by ablation before any component is credited.
- **Inconclusive ≠ negative.** If a change lands inside the noise band *and* Phase 0
  validation still doesn't track 2024, the result is **inconclusive** — you learned
  nothing about the change, only about the measurement. Do not KEEP or CUT on an
  inconclusive result; fix measurement or escalate to the ensemble (§Phase 5) and
  re-test.

### 1.5 Negative results are results

Every PARK/CUT gets a one-paragraph Decision Log entry: hypothesis, what was built,
how measured, outcome, and *why* (mechanistic guess). These entries are thesis
material — a rigorous account of what does **not** move G×E prediction on this
dataset is a legitimate contribution, and it is the backbone of the fallback in §3.

### 1.6 Risk ordering of the roadmap

The phases are deliberately ordered **cheap/high-confidence first, expensive/
speculative last.** If the compute or calendar budget runs out mid-plan, the safe
wins (turn on parent embeddings, fix the tokenizer, fix the env-PCC gradient) are
already banked, and only the speculative architecture work is lost.

---

## 2. Development plan (phased)

Each phase states a **hypothesis**, what to **build**, the **pre-registered bar**,
**if it fails**, and the **keep/cut default**. Phases are gated: do not start *n+1*
until *n*'s decisions are logged.

### Phase 0 — Make the result measurable (PREREQUISITE; blocks everything)

- **Hypothesis.** A validation scheme that mimics 2024 (novel parent1 × known tester,
  novel year) will rank configs more faithfully than LEO, and a seed/fold ensemble
  harness will let us read small deltas through the noise.
- **Build.** (1) **Cross-Tester / Leave-Year-and-Genotype-Out (LYGO)** validation:
  hold out hybrids whose parent1 is unseen but whose tester is in {PHP02, LH287}, in a
  held-out year — the only split that matches the 2024 genetic+temporal task (see
  Strategy D in `analyze_holdout_strategies.py`). (2) A **seed-ensemble + fold harness**
  that reports mean ± σ for any config in one command. (3) Keep LEO as a secondary
  diagnostic, not the selector.
- **Bar.** Validation–test rank correlation across a handful of known runs visibly
  better than LEO's; harness produces ±σ bands automatically.
- **If it fails.** If no holdout tracks 2024 (a real possibility per the diagnostics),
  fall through to **ensemble-over-configs** as the selection-free strategy (Phase 5)
  and treat all single-config deltas as inconclusive.
- **Default.** KEEP (this is infrastructure; it stays regardless of headline).

### Phase 1 — Feature signal (highest expected value)

- **Hypothesis.** The 0.42→0.45 gap is dominated by *missing input signal*: season-
  aggregated covariates destroy timing-of-stress (the physical basis of G×E), and the
  most test-relevant feature — parental identity — is currently disabled.
- **Build.**
  1. **Temporal weather / envirotyping encoder.** Pull raw daily weather per
     environment from NASA POWER by (lat, lon, planting→harvest) — coordinates already
     exist in preprocessing. Compute biologically-meaningful series (GDD, VPD, ET0,
     P−ETP, FRUE; cf. EnvRtype) and encode the `(T_days, n_vars)` sequence (small
     temporal transformer or 1D-conv/SSM) into an env-temporal embedding concatenated
     with — or replacing — the aggregated covariates. **This is the single change most
     likely to clear 0.45 because it adds new information.** See §5.1.
  2. **Per-feature tokenizer (FT-Transformer style).** Replace the shared
     `nn.Linear(1, n_embd)` env projection and shared `vocab=3` marker embedding so
     feature/locus identity lives in *content*, not only in the sinusoidal code. See
     §5.2. *(Correctness fix — keep on principle.)*
  3. **Enable existing features.** Turn on `use_parent_embeddings` (P1/P2 GCA tokens —
     directly targets novel-cross transfer) and `use_dual_channel` (additive +
     dominance — dominance is hybrid vigor). Both are implemented and off in the best
     run; ablate each.
- **Bar.** Per change, +0.005 env-PCC over baseline beyond the ±σ band, on the Phase 0
  selector.
- **If it fails.** Weather null → **PARK** behind a flag (re-test after Phase 4 depth;
  a 1-block model may be unable to exploit a rich temporal embedding). Parent/dual-
  channel null → PARK + document (suggests GCA is already implicit in the marker
  tokens). Tokenizer is a correctness fix → **KEEP** regardless.
- **Default.** KEEP tokenizer; KEEP features that clear the bar, else PARK.

### Phase 2 — Objective & optimization (cheap, stabilizing)

- **Hypothesis.** The env-PCC gradient is computed on too-small effective batches and
  the auxiliary "contrastive" loss is weak; fixing both lowers gradient variance and
  sharpens within-env ranking.
- **Build.**
  1. **Differentiable cross-GPU env-PCC.** `utils/loss.py:envwise_pcc` uses *local*
     sufficient statistics ("no all-reduce to preserve gradients"); with microbatch
     256 and 32/env, each `r_e` is over ~32 samples on ~8 envs/GPU. All-gather the
     per-env stats `(count, Σx, Σy, Σxx, Σyy, Σxy)` with `torch.distributed.nn` (the
     gather is differentiable) and form `r_e` over the full 8192 batch. See §5.3.
     *(Correctness fix.)*
  2. **Fisher-z training ablation.** Keep plain macro-r for *reporting* (matches the
     metric), but test a Fisher-z'd *training* loss so a few near-±1 environments
     don't dominate the gradient.
  3. **Soft-rank auxiliary.** Add a small `envspearman` / ListMLE term (already have
     differentiable soft ranks) to the env-PCC objective.
- **Bar.** Lower run-to-run σ for (1); +0.005 for (2)/(3).
- **If it fails.** (1) KEEP regardless (correct gradient over a larger sample is the
  right computation). (2)/(3) CUT if flat — these are objective tweaks, not
  information.
- **Default.** KEEP (1); KEEP (2)/(3) only if they clear the bar.

### Phase 3 — Representation pretraining (leverages the full dataset)

- **Hypothesis.** The encoders are under-trained by a 0.42-PCC supervised signal;
  self-/cross-supervision on the *full* panel (incl. unlabeled inbreds) yields better
  initializations.
- **Build.**
  1. **Marker MLM.** BERT-style masked-dosage modeling (mask ~15% of loci, predict
     from context) on all genotypes, learning LD structure; fine-tune on yield. Uses
     data the supervised loss currently ignores — the "massive dataset" lever.
  2. **Cross-modal CLIP (G↔E).** Replace the current intra-modal kernel alignment with
     a genuine cross-modal InfoNCE: an observed plot `(genotype, environment)` is a
     positive pair; mismatched (genotype, env) pairs are negatives. This learns the
     G×E manifold directly (which genotypes thrive where). See §5.4. Pretrain encoders,
     then fine-tune.
- **Bar.** Improvement in the **fine-tuned** metric, not the pretext loss.
- **If it fails.** PARK the recipe; retain a checkpoint if cheap. A CLIP null is
  informative (the batch may lack enough shared genotypes across envs for useful
  negatives) — log it.
- **Default.** PARK unless fine-tuned metric clears the bar.

### Phase 4 — Architecture for depth (speculative; do last)

- **Hypothesis.** One block wins today because 2727 flat tokens under O(n²) attention
  make depth unaffordable/unstable; reducing the token burden unlocks depth that can
  finally exploit Phase 1–3 signal.
- **Build.**
  1. **Perceiver latent bottleneck.** Cross-attend 2727 inputs into 32–64 latents,
     run a deep stack on the latents, read out from a learned latent. Buys many layers
     of G×E mixing cheaply.
  2. **Factorized G×E.** Encode G and E separately; model interaction explicitly via
     cross-attention / FiLM (E modulates G) — mirrors `y = μ + E + G + G×E` from
     `fit_decomposition.py` and makes the interaction inspectable.
  3. **Mamba/SSM on the *ordered* tracks only.** Use selective SSMs on the **marker
     sequence** (real 1D order → LD; linear-time, enables depth over all 2024 markers)
     and on the **temporal weather** stream (§1). *Not* on the unordered env-feature
     set — SSMs exploit order the env tokens don't have.
- **Bar.** Clears +0.005 *and* the complexity is justified vs. the simplest config
  that matches it.
- **If it fails.** **CUT.** Architecture that doesn't pay for its complexity is debt.
- **Default.** CUT unless clearly justified.

### Phase 5 — Classical floor + blend (also the honest fallback)

- **Hypothesis.** A reaction-norm/GBLUP model captures linear G+E+G×E signal the
  transformer may under-fit; blending recovers a reliable fraction of a point and
  sidesteps config-selection.
- **Build.** A GBLUP / reaction-norm baseline (Jarquín-style EC×marker covariance,
  optionally enviromic kernels from EnvRtype) and an **ensemble** of the best
  transformer configs + the GBLUP, weighted on the Phase 0 selector (or simple
  averaging if nothing tracks). This is the selection-free strategy flagged as
  "Strategy F" in `analyze_holdout_strategies.py`.
- **Bar.** Ensemble env-PCC ≥ best single model + its σ.
- **If it fails.** Ensembling rarely *hurts* macro-PCC; if it does, fall back to the
  single best calibrated model.
- **Default.** KEEP the blend as the submission artifact.

---

## 3. If the full plan does not reach 0.45

The plan is built to be **robust to its own failure.** Two backstops:

1. **A reliable floor.** Phase 5 (ensemble + GBLUP blend) almost always recovers a
   fraction of a point over any single model, and the affine-calibration head already
   secures competitive RMSE independently of PCC. So the *downside* is bounded near
   the current 0.42, not below it.

2. **A defensible thesis result either way.** If 0.45 proves out of reach after the
   roadmap, the contribution becomes a **rigorous characterization of why** — the
   validation–test structural mismatch (§0.2) and the novel-cross combining-ability
   ceiling (§0.1), backed by `analyze_holdout_strategies.py` /
   `analyze_val_failure.py` and the full Decision Log of measured nulls (§1.5). "Here
   is the barrier and what it would take to break it" is a legitimate, publishable
   outcome, and the literature (§6) supports that combining-ability + high-resolution
   envirotyping is the established lever. Hitting the number is the goal; explaining
   the number is the fallback — and both are wins.

**What is *not* an acceptable response to failure:** silently deleting parked
features, tuning hyperparameters against a non-tracking validation set until the
metric happens to move (that is fitting noise), or reporting a single lucky seed.

---

## 4. Decision Log (append-only)

> One entry per experiment. Template:
>
> ```
> ### YYYY-MM-DD — <short name>  [KEEP | ITERATE | PARK | CUT]
> Class:        correctness-fix | additive-feature | arch-swap | pretraining
> Hypothesis:   …
> Built:        … (commit / flag)
> Measured:     split, #seeds, mean ± σ vs baseline 0.42321
> Outcome:      Δenv-PCC = …  (in/out of noise band)
> Decision +    KEEP/ITERATE/PARK/CUT because …
>   rationale:  (mechanistic guess if null)
> ```

*(No entries yet — begin at Phase 0.)*

---

## 5. Code appendix (reference skeletons — adapt to live `config`/signatures)

These match the existing `models/model.py` / `utils/loss.py` structure (config-driven,
flag-gated). They are starting points, not drop-in final code.

### 5.1 Temporal weather encoder

```python
class WeatherEncoder(nn.Module):
    """Encode a per-environment daily weather sequence into one env-temporal token.
    Input:  w (B, T_days, n_wvars)  e.g. GDD, VPD, ET0, P-ETP, FRUE, Tmin/Tmax, precip
    Output: (B, n_embd)
    """
    def __init__(self, n_wvars, n_embd, n_layers=2, n_heads=4, dropout=0.15, kind="transformer"):
        super().__init__()
        self.in_proj = nn.Linear(n_wvars, n_embd)
        self.kind = kind
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(n_embd, n_heads, 4 * n_embd,
                                               dropout=dropout, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, n_layers)
        elif kind == "ssm":
            # Mamba/S4 over the time axis — natural fit for an ordered series.
            self.body = build_ssm_stack(n_embd, n_layers)   # external dep
        self.pool = nn.Linear(n_embd, n_embd)               # attention-pool or mean

    def forward(self, w):
        h = self.in_proj(w)                 # (B, T, C)
        h = self.body(h)                    # (B, T, C)
        return self.pool(h.mean(dim=1))     # (B, C)  — swap mean for attention pool
```
Integration: append the returned embedding as an extra token in `_encode`, gated by a
`config.use_weather_seq` flag, alongside (not instead of) the aggregated covariates
until the ablation says otherwise.

### 5.2 Per-feature tokenizer (FT-Transformer style)

```python
class FeatureTokenizer(nn.Module):
    """Per-feature affine: token_i = v_i * W_i + b_i. Identity in content, not just PE."""
    def __init__(self, n_features, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, n_embd)); nn.init.normal_(self.weight, std=0.02)
        self.bias   = nn.Parameter(torch.zeros(n_features, n_embd))

    def forward(self, x):                      # x: (B, n_features)
        return x.unsqueeze(-1) * self.weight + self.bias   # (B, n_features, n_embd)
```
Use for the 702 env scalars; for markers, prefer a per-locus embedding table (or add a
learned locus-position embedding) over the single shared `vocab=3` lookup.

### 5.3 Differentiable cross-GPU env-PCC

```python
def envwise_pcc_ddp(pred, target, env_id, world_size, eps=1e-8, min_samples=4):
    """Per-env Pearson over the FULL global batch by all-gathering sufficient stats.
    Keeps gradients (uses the differentiable all_gather)."""
    import torch.distributed.nn as distnn
    pred, target = pred.squeeze(-1).float(), target.squeeze(-1).float()
    E = int(env_id.max().item()) + 1
    acc = lambda v: v.new_zeros(E).scatter_add(0, env_id, v)
    local = torch.stack([acc(torch.ones_like(pred)), acc(pred), acc(target),
                         acc(pred**2), acc(target**2), acc(pred*target)])      # (6, E)
    g = torch.stack(distnn.all_gather(local)).sum(0)                            # (6, E) summed over ranks
    n, sx, sy, sxx, syy, sxy = g
    n = n.clamp_min(1.0)
    cov  = sxy/n - (sx/n)*(sy/n)
    vx   = (sxx/n - (sx/n)**2).clamp_min(eps)
    vy   = (syy/n - (sy/n)**2).clamp_min(eps)
    r    = (cov / (vx.sqrt()*vy.sqrt() + eps)).clamp(-1, 1)
    valid = g[0] >= min_samples
    return 1.0 - r[valid].mean() if valid.any() else (pred.sum()*0.0 + 1.0)
```

### 5.4 Cross-modal CLIP loss (G↔E)

```python
class CrossModalCLIP(nn.Module):
    """InfoNCE over (genotype, environment) pairs from observed plots.
    Diagonal = true co-occurrences (positives); off-diagonal = mismatched (negatives)."""
    def __init__(self, temp=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1/temp))))

    def forward(self, g_emb, e_emb):                      # (B, d) each, projected
        g = F.normalize(g_emb, dim=-1); e = F.normalize(e_emb, dim=-1)
        logits = self.logit_scale.exp() * g @ e.t()       # (B, B)
        labels = torch.arange(g.size(0), device=g.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
```
Caveat to test: needs enough distinct genotypes per batch for meaningful negatives;
env-stratified batching already helps. If negatives are too easy (different envs are
trivially separable), mine harder negatives within tester groups.

---

## 6. Literature backing

### G×E genomic prediction, combining ability, envirotyping
- **Jarquín, D., Crossa, J., Lacaze, X., et al. (2014).** A reaction norm model for
  genomic selection using high-dimensional genomic and environmental data. *Theor.
  Appl. Genet.* 127(3):595–607. doi:10.1007/s00122-013-2243-1. — Foundational
  marker×EC reaction-norm model; justifies factorized/explicit G×E (Phase 4.2) and
  the reaction-norm baseline (Phase 5).
- **Jarquín, D., de Leon, N., Romay, C., et al. (2021).** Utility of climatic
  information via combining ability models to improve genomic prediction for yield
  within the Genomes to Fields maize project. *Front. Genet.* 11:592769.
  doi:10.3389/fgene.2020.592769. — *Directly on this dataset:* climate + combining-
  ability lifts G2F yield prediction; core support for Phase 1 (weather) and the
  combining-ability framing in §0.
- **Costa-Neto, G., Galli, G., Carvalho, H.F., Crossa, J., Fritsche-Neto, R. (2021).**
  EnvRtype: a software to interplay enviromics and quantitative genomics in
  agriculture. *G3* 11(4):jkab040. doi:10.1093/g3journal/jkab040. — Envirotyping
  pipeline (NASA POWER → GDD/VPD/ET0/FRUE) and enviromic kernels; the concrete recipe
  for Phase 1's derived weather features.
- **Costa-Neto, G., Fritsche-Neto, R., Crossa, J. (2021).** Nonlinear kernels,
  dominance, and envirotyping data increase the accuracy of genome-based prediction in
  multi-environment trials. *Heredity* 126:92–106. doi:10.1038/s41437-020-00353-1. —
  Supports the dominance channel (Phase 1.3) and nonlinear E modeling.
- **VanRaden, P.M. (2008).** Efficient methods to compute genomic predictions. *J.
  Dairy Sci.* 91(11):4414–4423. — GRM definition used in `compute_grm_similarity`.
- **Meuwissen, T.H.E., Hayes, B.J., Goddard, M.E. (2001).** Prediction of total
  genetic value using genome-wide dense marker maps. *Genetics* 157(4):1819–1829. —
  Genomic-selection / GBLUP foundation behind Phase 5.

### Deep learning for tabular, sequence, and genomic prediction
- **Gorishniy, Y., Rubachev, I., Khrulkov, V., Babenko, A. (2021).** Revisiting Deep
  Learning Models for Tabular Data (FT-Transformer). *NeurIPS.* — Per-feature
  tokenization (Phase 1.2 / §5.2).
- **Jaegle, A., et al. (2021).** Perceiver: General Perception with Iterative
  Attention (*ICML*); Perceiver IO (companion). — Latent bottleneck for depth at low
  token cost (Phase 4.1).
- **Gu, A., Dao, T. (2023).** Mamba: Linear-Time Sequence Modeling with Selective
  State Spaces. — SSM on ordered marker/weather tracks (Phase 4.3); see also **Gu,
  Goel, Ré (2022)**, S4, *ICLR.*
- **Vaswani, A., et al. (2017).** Attention Is All You Need. *NeurIPS.* — Base
  encoder.
- **Shazeer, N., et al. (2017).** Outrageously Large Neural Networks: the Sparsely-
  Gated Mixture-of-Experts Layer. *ICLR.* — MoE FFN (current best config).
- **Montesinos-López, O.A., Montesinos-López, A., Crossa, J., et al. (2021).** A
  review of deep learning applications for genomic selection. *BMC Genomics.* —
  Survey context for DL-vs-classical in this domain.

### Pretraining & contrastive objectives
- **Devlin, J., et al. (2019).** BERT: Pre-training of Deep Bidirectional Transformers.
  *NAACL.* — Masked-token pretraining → marker MLM (Phase 3.1).
- **He, K., et al. (2022).** Masked Autoencoders Are Scalable Vision Learners. *CVPR.*
  — High-mask-ratio masked pretraining; alternative MLM framing.
- **Radford, A., et al. (2021).** Learning Transferable Visual Models From Natural
  Language Supervision (CLIP). *ICML.* — Cross-modal contrastive → G↔E CLIP
  (Phase 3.2 / §5.4).
- **van den Oord, A., Li, Y., Vinyals, O. (2018).** Representation Learning with
  Contrastive Predictive Coding (InfoNCE). — Contrastive objective behind §5.4.
- **Chen, T., et al. (2020).** A Simple Framework for Contrastive Learning of Visual
  Representations (SimCLR). *ICML.* — Projection-head pattern already used in
  `g_contrast_proj`.

### Ranking objectives & training
- **Blondel, M., et al. (2020).** Fast Differentiable Sorting and Ranking. *ICML.* —
  Differentiable soft-rank losses (Phase 2.3; existing `envspearman`/`ktau`).
- **Loshchilov, I., Hutter, F. (2019).** Decoupled Weight Decay Regularization
  (AdamW). *ICLR.* — Optimizer in use.
- **Huang, G., et al. (2016).** Deep Networks with Stochastic Depth. *ECCV.* —
  Stochastic depth in the encoder.
- Fisher z-transformation (R.A. Fisher, 1915/1921) — variance-stabilizing transform
  motivating the Phase 2.2 training-loss ablation.

### Crop yield / weather-sequence modeling (see also)
- **Messina, C.D., Technow, F., Tang, T., et al. (2018).** Leveraging biological
  insight and environmental variation to improve phenotypic prediction: integrating
  crop growth models with whole-genome prediction. *Eur. J. Agron.* — Crop-growth-
  model-derived stress features as inputs; conceptual backing for envirotyping over
  raw season aggregates.
- **van Klompenburg, T., Kassahun, A., Catal, C. (2020).** Crop yield prediction using
  machine learning: a systematic literature review. *Comput. Electron. Agric.* —
  Survey of weather-driven yield models.
- **Khaki, S., Wang, L. (2019).** Crop yield prediction using deep neural networks.
  *Front. Plant Sci.* — DNN yield prediction precedent.
