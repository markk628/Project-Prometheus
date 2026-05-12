# v6 Dev Log

Working notes from the v6 daily-bar SAC trading project. Architectural and
structural experiments built on the v5 run-5 baseline (UTD=2, slow alpha LR
1e-5, all run-3 regime features, all run-2 data fixes).

For the v5 history see `dev_log_v5.md`. For the original v6 plan see
`v6_handoff.md`.

---

## Determinism Hardening (pre-run-1 infrastructure)

Originally on the v6_handoff list as a "do alongside" item. Pulled forward
as the first thing because run 1's results came out ambiguous and we needed
a clean methodology before deciding what to do about it.

**Changes shipped:**

```python
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Placed at the top of `trainer.py::main` before any tensor hits CUDA.

**What it actually fixes:** GPU kernel selection and reduction-order
nondeterminism. The pre-existing three-line seed (random + numpy + torch)
covers language-level RNGs but not the cuDNN algorithm-selection layer or
parallel-reduction order. Without `cudnn.deterministic=True`, two runs of
identical config can diverge at the bit level via different conv kernel
choices, which compounds over 1800 episodes into materially different
end states.

**Cost:** ~10-15% wall-clock hit on this network (encoder dominated by
transformer; conv branch is small). Acceptable.

**Verification:** ran v5-with-determinism and v6-with-determinism with
SEED=42 (matching the pre-determinism runs). Validation summaries came out
bit-identical to the previous runs in both cases — meaning the
pre-determinism runs were already getting lucky with consistent kernel
choices for this specific network shape, NOT that determinism wasn't
working. Determinism is correctly enabled; it just turned out cuDNN
nondeterminism wasn't the dominant noise source for this setup. The
dominant variance source is the seed itself (init, replay sampling,
exploration), which is what motivated the MC sweep below.

**Process lesson:** "results matched after enabling determinism" is
ambiguous on its own — could mean either "determinism wasn't doing
anything" or "determinism is on but cuDNN noise wasn't a big factor here."
Disambiguate by varying the seed: if results barely move across seeds, the
network is robust to all stochasticity; if they move noticeably, seed
matters and determinism is your reproducibility floor for that variance.

---

## Run 1 — Remove Sharpe from portfolio_state (REJECTED)

**Hypothesis:** Per-step Sharpe in `portfolio_state` is a footgun. It's
clipped to [-3, 3] in `_get_observation`, which makes "real high
performance" and "degenerate near-zero activity" indistinguishable to the
network when both saturate at the upper bound. Could reinforce undertrading.
Sharpe stays in `_get_info` for logging — only removed from observation.

**Changes:**

- `environment.py::_get_observation`: dropped Sharpe slot from
  `portfolio_state`. 7 dims → 6 dims (cash ratio, stock ratio, unrealized
  PnL, hold-time ratio, current drawdown, win rate).
- `trainer.py::main`: replay buffer + agent constructed with
  `portfolio_state_len=6`.
- Networks (Actor, Critic) take `portfolio_state_len` as constructor arg
  already, no hardcoded dim changes needed.
- Replay buffer flat-state width updated automatically via passed-in arg.
- Breaks checkpoint compatibility with v5 — clean v6 break.

**Initial result (single seed = 42):**

| Metric | v5 | v6 | Δ |
|---|---|---|---|
| Validation return mean | +8.52% | +7.69% | -0.83 |
| Validation return std | 10.97 | 13.63 | +2.66 |
| Validation Sharpe mean | 0.47 | 0.48 | +0.01 |
| Validation Sharpe median | 0.38 | 0.45 | +0.07 |
| Validation positive rate | 74.4% | 71.7% | -2.7pt |

Genuinely ambiguous. Mean Sharpe slightly up, median Sharpe up more
clearly, but return mean down and std up. Single fold (fold 7, COVID) was
doing all the work in the negative direction — v5 fold 7 mean = -3.18%, v6
fold 7 mean = -16.84%. Excluding fold 7, v6 was unambiguously better
across the rest. Two competing readings:

1. **Signal:** Sharpe-in-obs was acting as a stability anchor through
   regime shifts. Removing it made COVID handling fragile.
2. **Noise:** seed-level variance, fold 7 just happens to be where this
   particular run's noise landed.

Couldn't tell from one run. Pulled forward determinism hardening + MC
sweep to disambiguate.

**MC sweep methodology:**

- N=3 seeds per condition: 42, 43, 44.
- v5 baseline (Sharpe-in-obs) and v6 (Sharpe-removed) both rerun with all
  3 seeds.
- Determinism on for all 6 runs (cudnn.deterministic=True,
  cudnn.benchmark=False, all seeds covered) — varies algorithmic
  stochasticity across seeds, holds hardware-noise floor constant.
- 6 runs total, ~54 wall-clock hours.

**MC results — same-seed paired comparison (v6 minus v5):**

| Seed | Ret Δ | Sharpe Δ |
|---|---|---|
| 42 | -0.84 | +0.01 |
| 43 | -1.30 | -0.03 |
| 44 | -2.05 | -0.07 |

All three seeds: v6 has lower return. Two of three: v6 has lower Sharpe.
The seed-42 result that originally looked ambiguous was the best of the
three; seeds 43 and 44 both came out worse on both metrics.

**Fold-level head-to-head across 27 paired comparisons (3 seeds × 9 folds):**

- Return: v6 beats v5 in 11 of 27, loses in 16
- Sharpe: v6 beats v5 in 12 of 27, loses in 15

Consistent ~60% loss rate across folds. Not a coin flip — a small but
real disadvantage everywhere.

**Fold 7 across all seeds (v6 - v5 paired):**

| Seed | Ret Δ | Sharpe Δ |
|---|---|---|
| 42 | -13.65 | -0.29 |
| 43 | -2.37 | -0.14 |
| 44 | -4.16 | -0.14 |

v6 fold 7 is worse than v5 fold 7 on every seed. Magnitude varies (seed
42 was the worst case) but direction is unanimous. Fold 7 collapse is
NOT seed lottery — it's a real Sharpe-removal effect on COVID handling.

**Excluding fold 7:**

| | Return mean | Sharpe mean |
|---|---|---|
| v5 (3-seed) | +10.43% | +0.53 |
| v6 (3-seed) | +9.70% | +0.52 |

Even discounting the COVID fold, v6 is marginally worse on both axes.

**Diagnosis:** Sharpe-in-obs apparently was doing useful work as a
"is my recent behavior working" trajectory signal that helped the policy
not panic during regime breaks. The clipping footgun was real in
principle — high-performance and degenerate-low-activity DO map to the
same bound — but in practice the network was extracting useful
recent-performance context from the unclipped middle of the distribution
and only saturating at the bound infrequently enough that it didn't
dominate. Removing the dim removed a useful prior, exposed by the
fragility on the hardest fold.

**Decision: REVERT.** Restore Sharpe to `portfolio_state`. v6 baseline
returns to v5 run-5 config exactly (7 portfolio dims). Future runs build
on that.

**Negative result, useful diagnosis** — same shape as v5 run 4 (UTD=4
overfit). The change was well-motivated, the test was rigorous, and the
data says don't ship it.

---

## Cross-Cutting Lessons from Run 1

**Single-seed runs lie about direction.** v6 seed 42 alone gave
"marginal win" on Sharpe. Three-seed MC gave "consistent loss across
seeds." If we'd shipped run 1 as a win and built the rest of v6 on top,
every subsequent ablation would have been measured against a
silently-inferior baseline. Compounding wrong calls for the rest of v6.

**MC methodology is now standard for v6 ablations.** N=3 seeds (42, 43,
44) per condition going forward. 3× wall-clock cost is real but pays for
itself the moment it prevents one bad direction-call. Already paid for
itself once with run 1.

**Save the v5 reference distribution.** 3-seed v5 baseline is now the
real reference: ret_mean ~+9.2%, sh_mean ~+0.48, with cross-seed return
spread of ~6 pts. Every future v6 result compares against this
distribution, not against any single v5 run number. Numbers from
`dev_log_v5.md` (the +8.52% / 0.47 v5 run-5 result) are one sample, not
the population.

**Determinism + seed sweep is the right pairing.** Determinism alone gives
reproducibility. Seed sweep alone gives variance estimates but is muddied
by hardware-noise floor. Both together: clean variance-on-algorithmic-
stochasticity estimates. The version-agnostic standard for this project
from here on.

**"Real footgun in principle" doesn't mean "footgun in practice."** The
Sharpe-clipping argument was correct that the saturation makes
high-perf and no-activity indistinguishable AT the bound. What it didn't
account for: how often the metric actually hits the bound vs. spends time
in the middle of the distribution carrying real information. Real-data
test was the right call.

**Reading 2am intuitions correctly.** The original framing for run 1 was
"this is a real footgun, removing it should help or at worst be neutral."
Reality was "this is a real footgun in principle but the network was
using it productively anyway." Worth noting because the same logic
pattern probably applies to other "this is obviously bad design"
intuitions about the observation space. Test before shipping.

---

## Run 2 — Remove win_rate from portfolio_state (REJECTED)

**Hypothesis:** Per-episode win_rate is bimodal at the episode level. The
trainer's `_get_win_rate` only counts a "completed trade" when shares_held
goes to zero (full exit) or at EOD liquidation. With the dollar deadband
at 7% and the agent settling into a target-exposure mode, most episodes
complete 0-1 round-trips. One round-trip → win_rate ∈ {0%, 100%}. Train
median for win_rate hovered at 0%, and the metric was structurally
asking too much of one episode given how the env counts trades. The
training-curve plot (`win_rates.png`) showed exactly this bimodal
pathology — train MA(10) bouncing between extremes, validation smoothed
to multiples of 10 around 50%. Pre-staked TODO from the v5 portfolio_state
design.

Premise: removing win_rate should be neutral or positive — the dim is
mostly noise, the network is conditioning policy on a low-SNR signal.

**Changes:**

- `environment.py::_get_observation`: dropped win_rate slot from
  `portfolio_state`. 7 dims → 6 dims (cash ratio, stock ratio,
  unrealized PnL, sharpe, hold-time ratio, current drawdown).
- win_rate stays in `_get_info` for logging.
- v6 baseline for run 2 = v5 run-5 config exactly (Sharpe restored after
  run 1 rejection).

**Methodology:** 3-seed MC from the start (42, 43, 44). No single-seed
run first — run 1 taught us not to.

**Results — same-seed paired comparison (v6-no-WR minus v5-baseline):**

| Seed | Ret Δ | Sharpe Δ | Pos% Δ |
|---|---|---|---|
| 42 | -0.47 | -0.025 | -2.00 |
| 43 | -1.04 | -0.035 | +0.06 |
| 44 | -5.02 | -0.120 | -5.50 |

All three seeds: v6 worse on return AND Sharpe. Direction unanimous,
magnitudes range from "barely measurable" to "substantial." Seed 44
collapsed -5pt return / -0.12 Sharpe — bigger paired regression than any
single seed showed during run 1.

**Cross-seed aggregate:**

| | Return mean | Sharpe mean |
|---|---|---|
| v5 baseline (3-seed) | +9.18% ± 3.41 | +0.48 ± 0.05 |
| v6 no-WR (3-seed) | +7.01% ± 1.64 | +0.42 ± 0.02 |

**Fold-level head-to-head across 27 paired comparisons:**

- Return: v6 wins 9, losses 18 (33% win rate)
- Sharpe: v6 wins 10, losses 17 (37% win rate)

Worse than run 1 (which had ~41-44% win rates). win_rate-removal is more
consistently degrading across folds than Sharpe-removal was.

**Per-fold pattern is informative:**

```
Fold 6: Δ ret -8.23%,  Δ sh -0.13   ← peak fold takes a big hit
Fold 7: Δ ret -5.91%,  Δ sh -0.14   ← COVID, hit again (same as run 1)
Fold 9: Δ ret -2.51%,  Δ sh -0.22   ← largest Sharpe drop
Others: neutral or small negatives
```

Fold 6 dropping is notable — it's not a regime-shift fold, it's where v5
has its best fold (peak +42.30% mean, the fold that first cleared
sustained Sharpe > 1.0 in v5 capstone). v6 dropped to +34.07% / 1.04
mean Sharpe. So unlike run 1 (Sharpe-removal mostly hurt regime-shift
handling), win_rate-removal hurts even in peak performance folds. The
network was using win_rate productively in MORE situations than Sharpe.

**Excluding fold 7:**

| | Return mean | Sharpe mean |
|---|---|---|
| v5 baseline | +10.43% | +0.53 |
| v6 no-WR | +8.72% | +0.48 |

Still consistently worse. Not a fold-7 phenomenon this time — the
regression is broad-based.

**Diagnosis:** Despite the bimodal-at-episode pathology being real, the
network was using win_rate productively. Probably as a slow-moving
"recent strategy quality" signal — even with episode-level bimodality,
the trajectory of win_rate across batches gives the policy useful
context about whether its current behavior is paying off. The bimodality
made the *training curve plot* look weird, but it didn't make the
*signal to the network* useless.

**Decision: REVERT.** Restore win_rate to `portfolio_state`. v6 baseline
returns to v5 run-5 config exactly (7 portfolio dims) for the second
time.

---

## Cross-Cutting Lessons from Run 2

**Two-for-two on portfolio_state subtractive ablations being clean
negatives.** Both Sharpe (run 1) and win_rate (run 2) looked like
"obvious footguns" by their math characteristics — clipping saturation,
bimodal episode-level distribution. Both turned out to be doing real
work. The pattern is now strong enough to act on.

**The "find footguns by inspection" approach to portfolio_state is
unreliable.** Math-level pathologies (clipping, bimodality, redundancy)
don't predict information-content-level usefulness. The network
extracts whatever signal is present in whatever distribution shape, and
"this looks ugly to a human" isn't the right test.

**Subtractive sweep stops here.** Two clean negatives is enough evidence.
hold_time_ratio was the planned next test — skipping it. Cost-benefit
flipped: ~27 more compute-hours for a test that's now low-EV given the
established pattern. Documenting the decision so future-me doesn't
relitigate.

**Future portfolio_state work pivots to additive only.** Cumulative
trajectory dims (realized P&L, fraction-time-invested, max-position),
windowing over portfolio_state — these add new info that's plausibly
orthogonal to existing dims, not subtract dims that "look bad."
Different research question, different EV profile.

**MC paid for itself again.** Seed 42 alone showed -0.47% return / -0.025
Sharpe — a "tiny shrug" result that could plausibly have been called
neutral and the change shipped. Seed 44 showed -5.02% / -0.12. Without
3-seed MC, ship-vs-revert call would have been wrong roughly 1-in-3 times
on this kind of marginal change. Methodology earned its keep.

---

## Run 3a — More regime features (REJECTED)

**Setup.** Three preprocessing-level changes shipped together as one variable
("more shared regime features"):

1. **File-split refactor of `feature_engineer.py`** — pulled the
   pre-class section out into 8 helper modules (`constants`, `sectors`,
   `nyse_calendar`, `splits`, `gaps`, `features`, `normalization`,
   `worker`). Pure code motion, verified bit-identical output via
   `df.equals` on unified.parquet. Not a behavior change.
2. **5 new macro regime feature families** in a new `regime_features.py`
   module:
   - Yield curve — `log(TLT/SHY)` smoothed at 5d / 60d + 60d delta
   - Credit spread — `log(HYG/LQD)` smoothed at 5d / 60d + 60d delta
   - Size factor — `log(IWM/SPY)` smoothed at 5d / 60d + 60d delta
   - Growth factor — `log(QQQ/SPY)` smoothed at 5d / 60d + 60d delta
   - Sector rotation — cross-sectional dispersion + topbottom of the 11
     XL\* ETF cumulative returns at 60d / 200d + 20d deltas on long
     horizons
   - **18 new shared regime columns total** (12 levels + 6 deltas).
3. **`_resolve_close_col` helper** for transparent handling of
   lifecycle-segmented regime tickers (HYG.2, QQQ.2 — the .1 segments
   got dropped by MIN_TICKER_LENGTH).

Also fixed a latent bug in `trainer.py::load_tickers_from_unified` where
the column classifier hardcoded `("breadth_", "vix_term_")` as the only
shared regime prefixes — all 18 new macro columns were silently dropped
from `TickerData.data` on the first attempted run. Found via the new
`TickerData.columns` attribute by inspecting AAPL in a notebook (data
shape was 92 instead of 110). Fix: unified prefix list in
`constants.SHARED_REGIME_PREFIXES`, imported by both auditor and trainer.

**Hypothesis.** Adding macro regime context should help generalization
across regime shifts, especially the COVID-like fold 7 fragility seen in
runs 1 and 2. The "broad-spectrum regime context" framing — yield curve
inversion, credit spread widening, sector dispersion — describes the
*mechanism* of regime change, not just specific events.

**Methodology.** 3-seed MC (42, 43, 44) with determinism on, comparing
against the 3-seed v5 baseline.

**Results — cross-seed aggregate:**

| | Return mean | Sharpe mean | Sharpe median | Pos rate |
|---|---|---|---|---|
| v5 baseline (3-seed) | +9.18% ± 3.41 | +0.485 ± 0.052 | +0.43 | 62.0% |
| v6 run 3a (3-seed)   | +7.79% ± 3.56 | +0.465 ± 0.063 | +0.46 | 59.9% |

**Same-seed paired delta (3a minus v5):**

| Seed | Δ Return | Δ Sharpe | Δ Pos% |
|---|---|---|---|
| 42 | +1.25 | +0.050 | -1.22 |
| 43 | -2.48 | -0.044 | -2.94 |
| 44 | -2.95 | -0.065 | -2.17 |

Unlike runs 1 and 2 (3/3 seeds unanimously worse), this is **1 seed win,
2 seed losses**. Closer to the noise floor than to a clean rejection.

**Per-fold breakdown (3-seed means, 3a minus v5):**

```
Fold   Δ ret    Δ sharpe    Interpretation
1      +0.28    +0.02       wash
2      +0.41    +0.05       small win
3      +0.77    +0.07       small win
4      -2.77    -0.06       meaningful loss
5      +2.70    +0.04       notable win  (regime-shift fold)
6      -6.78    -0.08       big loss     (v5's peak fold)
7      -6.13    -0.16       big loss     ← THE PROBLEM
8      +2.88    +0.03       notable win  (regime-shift fold)
9      -3.93    -0.08       meaningful loss
```

**Fold-level head-to-head (27 pairs):** 3a wins 13/27 on return, 15/27
on Sharpe. Coin-flip territory.

**Decision rule scorecard** (rule was set before MC kicked off):

| Criterion | Verdict |
|---|---|
| Cross-seed mean must improve return AND Sharpe | **FAIL** (-1.39 ret, -0.020 sh) |
| Regime-shift folds (5,7,8) must improve more than average | **FAIL** (-0.18 ret, -0.03 sh — flat) |
| Quiet-bull folds must NOT regress meaningfully | **MARGINAL** (-2.00 ret, -0.01 sh) |
| At least 6 of 9 folds improve | **FAIL** (5/9 on both ret and Sharpe) |

By the rule we agreed to before training kicked off, this rejects.

**The fold 7 result is the most important finding.** 3a was specifically
designed to test the hypothesis "macro regime context helps regime
shifts," and fold 7 (COVID) is the canonical regime-shift fold. 3a made
fold 7 *worse*: -6.13 pts return, -0.16 Sharpe, consistent across all 3
seeds. The hypothesis got falsified.

**The fold 5/8 vs fold 7 split is informative.** Folds 5 and 8 (other
regime-shift folds) *did* improve with macro features, by 2-3pts return.
But fold 7 sharply regressed. The likely mechanism: macro features
captured the shape of regime shifts that look like 2008 (gradual credit
spread widening, yield curve inversion, sector rotation) but COVID
didn't look like 2008. March 2020 was an "everything sells off including
treasuries briefly, then policy backstop" pattern that doesn't fit the
historical training data shape. The 60d-smoothed features were too slow
to register the shift; by the time they did, policy response had already
reshaped the regime.

**Fold 6 regression worth flagging separately.** v5's peak fold (the
first to sustain Sharpe > 1.0) dropped 6.78pts in return with 3a.
Mechanism likely: with new features available, the network reorganized
attention away from per-ticker signals that were carrying weight in fold
6. Adding features can hurt even when those features encode real
information, if they redirect the policy away from what was working.

**Decision: REVERT.** v6 baseline returns to v5 run-5 config for the
third time. Skip 3b — the speculative families (dollar, commodities,
international) have weaker priors than 3a's families and the bundle
hypothesis already failed at the strongest-prior version. Move on to
run 4.

---

## Cross-Cutting Lessons from Run 3a

**The single-variable rule applies even to bundled feature families.**
Run 3a shipped 5 feature families as one variable because they're all
"shared regime context" semantically. That's a defensible single-variable
boundary, but it means we can't attribute the result to individual
families. Folds 5/8 improved and fold 7 regressed — those might be
driven by different families within the bundle, and we don't know
which. Worth thinking about in future feature-batch runs: prefer
bundling families that share a *mechanism* (all credit-related, all
rate-related) over bundling families that just share a *category*
("macro stuff").

**Single-feature ablation within a rejected bundle is rarely worth the
cost.** Tempting to spend 5×3=15 runs (~135 hours) to find which family
helped vs hurt. Not worth it: (1) the bundle is already at the noise
floor, so individual family signals would be smaller still; (2) the
1-of-3 seeds positive pattern suggests no robust single-family signal
hiding inside; (3) the falsified hypothesis (fold-7 specifically) is the
informative result, not which family did it. Move forward instead of
rabbit-holing.

**Decision rule pre-commitment paid off.** Without the strict rule
written down before training, the temptation post-hoc would have been
to ship on "Sharpe median basically unchanged, two regime folds improved,
maybe okay overall." The rule kept us honest: the change failed on the
specific dimension it was designed to address (fold 7). Worth restating
the principle: **commit to the decision rule before MC results are in.**

**Speed-of-feature-response matters for the fold-7-style regime.** All
of run 3a's families use 60d smoothing. That's appropriate for slow
regime shifts (rate cycles, credit cycle turning) but too slow for sudden
shocks like COVID. If we ever come back to macro features, the design
question to think about is "what time scale of regime shift is this
feature targeting" — 5d / 20d short-window variants might catch what
60d misses, at the cost of more noise.

**Three rejected runs in a row is data about the problem shape, not the
methodology.**

- Run 1 (Sharpe removal): "obvious footguns" aren't always footguns
- Run 2 (win_rate removal): same lesson reinforced
- Run 3a (macro features): broad-spectrum regime context doesn't transfer
  to single-ticker decisions reliably

Each rejection has been *informative*, not just negative. Run 3a in
particular suggests something genuine: macro regime features describe
market-wide conditions, but the agent's decision is about whether to be
in *this specific ticker* right now. The marginal value of macro context
for that question is apparently small, because per-ticker idiosyncratic
noise dominates the single-ticker timing problem.

That's the v7 thesis confirming itself in v6 data: single-ticker timing
has a low ceiling. v5 run-5's +9.18% / 0.48 Sharpe (3-seed) is plausibly
close to the realistic ceiling for this approach. The bigger gains live
in v7's allocation pivot. If runs 4-7 also reject, that's not failure —
that's discovering the v6 architectural search space doesn't move the
needle, and the path forward really is v7.

---

## Where v6 Stands After Run 3a (Rejected)

- v6 baseline = v5 run-5 config. Both Sharpe and win_rate in
  portfolio_state (7 dims). No macro regime features.
- The file-split refactor of `feature_engineer.py` STAYS — that was pure
  code motion, verified bit-identical, and is an unambiguous codebase
  win. Future code goes into the right modules.
- `regime_features.py` and the `_resolve_close_col` helper stay in the
  codebase but aren't called from `_build_unified` anymore. Cheap
  insurance for the day macro features come back in a different form
  (v7 portfolio likely wants some of this work).
- The `SHARED_REGIME_PREFIXES` constant + the `TickerData.columns`
  attribute stay — those are quality improvements regardless of run-3a's
  fate. The trainer-classifier bug they exposed was real and would have
  silently affected any future regime-feature addition.
- v5 reference distribution unchanged: ret_mean +9.18% ± 3.41,
  sh_mean +0.48 ± 0.05.
- 3b (speculative regime features: dollar, commodities, international,
  factors not in 3a) is **shelved**, not scheduled. Justification:
  weaker priors than 3a, bundle hypothesis failed at the strongest
  version, and the international families need EFA/EEM data fixes
  anyway.

## Pending Run 4: Per-ticker architecture (encoder vs MLP)

Per v6_handoff: the current `FeatureExtractor` processes a (60, 25)
market-data sequence through CNN+Transformer, but many of the 25
features are already multi-horizon summaries themselves (log_return_5/
20/60, volatility_5_20_ratio, ema_5_20_ratio, etc.). Feeding 60
timesteps of multi-horizon summaries through a sequence encoder is
probably redundant with what the features already encode.

Two architectural options per the `networks.py` TODO:

1. **MLP-only market path.** Drop the encoder. Pass the last-timestep
   25-dim market vector through an MLP. **PREREQUISITE**: add per-ticker
   delta features in `feature_engineer.py` first (the 25 ticker features
   are levels and ratios, not changes — an MLP on a snapshot can't
   distinguish "building for 30 days" from "spiked yesterday").
2. **Minimal CNN + MLP.** Single 1D conv layer + AdaptiveAvgPool, no
   transformer. Cheaper insurance than option 1, no feature-engineering
   change needed.

Open question whether to ship the per-ticker delta features as Run 4a
(preprocessing only, no model change) before testing the MLP-only path
as Run 4b. The deltas are useful with the existing encoder too — they're
not strictly an MLP-only thing. Could be a clean two-run sequence.

3-seed MC, same standard.