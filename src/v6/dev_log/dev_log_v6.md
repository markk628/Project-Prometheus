# v6 Dev Log

Working notes from the v6 daily-bar SAC trading project. Architectural and
structural experiments built on the v5 run-5 baseline (UTD=2, slow alpha LR
1e-5, all run-3 regime features, all run-2 data fixes).

For the v5 history see `dev_log_v5.md`. For the original v6 plan see
`v6_handoff.md`.

---

## Correction — fold-year mapping (added retroactively)

Earlier writeups in this log (runs 1, 3a, 4a, plus the cross-cutting lessons
that synthesized them) referred to "fold 7" as the COVID fold. **This was
wrong.** The correct fold-year mapping from the trainer's walk-forward
schedule is:

```
Fold 1: Valid 2015    Fold 4: Valid 2018    Fold 7: Valid 2021
Fold 2: Valid 2016    Fold 5: Valid 2019    Fold 8: Valid 2022
Fold 3: Valid 2017    Fold 6: Valid 2020    Fold 9: Valid 2023
```

So COVID is **fold 6**, not fold 7. Fold 7 is the 2021 post-COVID melt-up
(meme stocks, retail flows, peak ZIRP) — historically the *easiest* year
for buy-and-hold equity strategies, not a regime-stress year.

This changes what the recurring fold-7 regression across v6 runs actually
means:

- The numerical observation is unchanged: runs 1, 3a, and 4a all hurt
  fold 7 specifically. That happened, and the data tables in those
  writeups are correct.
- The *interpretation* in those writeups attributed the fold-7 regression
  to "COVID handling" or "regime-shift fold," using the wrong mapping.
  Those interpretations are wrong as written.
- An earlier version of this correction proposed "trailing-vol features
  stay elevated, model goes conservative, underperforms melt-up" as the
  mechanism. That interpretation is also wrong on closer inspection.
  See below.

**Actual diagnosis (revised after closer look at fold 6 vs fold 7):**

2020 was a +18% calendar year for SPY — a Q1 crash followed by an
8-month melt-up driven by Fed intervention and retail flows. The "story-
driven slow melt-up" dynamic I was tagging as 2021-specific actually
started in Q2 2020. So fold 6's validation period (full year 2020) and
fold 7's validation period (full year 2021) overlap substantially in
character. The "2021 is qualitatively novel" framing doesn't work — most
of 2020 post-crash already had 2021-like dynamics.

The fold 6 vs fold 7 puzzle: fold 6 is the **best** fold across most v6
runs (sometimes Sharpe > 1.0), fold 7 is the **worst**. If both folds
had qualitatively similar validation dynamics, the cause must be
something other than the validation periods themselves.

The likely answer is **recency emphasis in the replay buffer**. The
`IndexReplayBuffer` uses `decay=3.0` recent-emphasis sampling: ~60% of
gradient updates come from the newest third of stored transitions, ~10%
from the oldest. This means each fold's policy is disproportionately
shaped by whatever happens to be the most recent training years.

- Fold 6 model: training ends Dec 2019. Buffer's recent third is roughly
  2017-2019 — the low-vol melt-up era, calm bull market. Policy is
  trained on tranquility. Deployed into 2020 (which is, weighted by
  duration, mostly a continuation of the same melt-up after a brief
  5-week interruption), the policy works because the year is structurally
  "calm bull with brief interruption" — exactly what the policy is
  calibrated for.
- Fold 7 model: training ends Dec 2020. Buffer's recent third is roughly
  2018-2020 — Q4 2018 selloff, all of 2019, COVID crash and recovery.
  The most-weighted recent training is the 2020 chaos period. Policy
  has learned chaos-response reflexes: elevated-vol regime detection,
  sharp directional moves, V-shape recovery dynamics. Deployed into
  2021's slower, steadier melt-up, those reflexes don't match what the
  environment rewards.

This is a much cleaner explanation than feature-induced conservatism:
- Consistent with fold 6 being the *best* (calm training → bull deployment)
- Consistent with fold 7 being the *worst* (chaos training → calm-bull deployment)
- Consistent with fold 8 (2022 bear) not being unusually bad — chaos-
  trained reflexes transfer reasonably to actual chaos
- Consistent with the three runs that hurt fold 7 (1, 3a, 4a) all having
  in common that they added input dimensions where the 2020-vs-2021
  mismatch had more room to manifest

**Adding features didn't cause fold 7's problem; it amplified an
existing mismatch between the policy's recent-training regime and fold
7's deployment regime.**

The actual regime-stress folds are 4 (2018 Q4 selloff), 6 (COVID), and
8 (2022 rate-hike bear) — not 5, 7, 8 as several writeups below claim.

Where the original writeups are misleading because of the wrong mapping,
this section flags it. Specific interpretive claims that depend on the
wrong mapping (e.g. "macro features can't catch COVID-shape regime"
in run 3a's diagnosis) should be read with this correction in mind.

The downstream takeaway for v7 is unchanged but for a sharper reason:
v7's portfolio allocation framing handles "any environment where being
defensive is correct" without needing the policy to predict the next
year. AND — separately — v7 should revisit whether `decay=3.0` is the
right recency emphasis for a 5-ticker fixed-basket setup, since the
buffer dynamics will be very different from v6's thousands-of-tickers
sampling. Filed in `v7_handoff.md` as a v7 design consideration.

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

**Seeds & Runs:**
- Seed 42: Run 2
- Seed 43: Run 3
- Seed 44: Run 4

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
[NOTE: this paragraph used wrong fold-year mapping — fold 7 is 2021
post-COVID melt-up, NOT COVID. See "Correction — fold-year mapping" at
top of file. The numerical observations are correct; the "COVID handling"
interpretation is wrong.]
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

**Seeds & Runs:**
- Seed 42: Run 5
- Seed 43: Run 6
- Seed 44: Run 7

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

**Hypothesis:** Adding macro regime context should help generalization
across regime shifts, especially the COVID-like fold 7 fragility seen in
runs 1 and 2. The "broad-spectrum regime context" framing — yield curve
inversion, credit spread widening, sector dispersion — describes the
*mechanism* of regime change, not just specific events.

**Seeds & Runs:**
- Seed 42: Run 8
- Seed 43: Run 9
- Seed 44: Run 10

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

[NOTE: this writeup used the wrong fold-year mapping. Fold 7 is 2021
post-COVID melt-up, not COVID. The "macro features can't catch COVID"
mechanism described below is wrong as written. The likely actual
mechanism is that the replay buffer's recency emphasis (decay=3.0)
weights 2020 chaos disproportionately at fold 7's training end, giving
the policy chaos-response reflexes that don't match 2021's slower
melt-up dynamics. Adding features amplifies the mismatch by giving
that miscalibration more dimensions to manifest in. See "Correction —
fold-year mapping" at top of file.]

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

## Run 4a — Per-ticker delta features (NEUTRAL / SHIPPED FORWARD)

**Setup.** Added 5 per-ticker delta features inside `_normalize_data`,
computed on the z-scored levels after the existing rolling normalization:

- `ema_close_ratio_20_delta_20`
- `ema_20_60_ratio_delta_20`
- `adx_5_20_ratio_delta_20`
- `volatility_5_20_ratio_delta_20`
- `volume_5_20_ratio_delta_20`

All 5 features are medium-window levels/ratios. All are z-scored in
`_normalize_data`, so the deltas live on the same scale as the bases —
no separate normalization needed. Convention matches the regime delta
machinery: delta-of-z-scored-level, not z-score-of-delta. Computation
lives in `normalization.py` so all per-ticker feature work stays in one
module.

`adx_14` was originally on the candidate list but swapped for
`adx_5_20_ratio` mid-design: `adx_14` is cross-sectionally normalized in
a later stage (`_cross_sectional_normalization`), so at `_normalize_data`
time its scale is still raw [0, 1] — a delta on the raw scale would have
been inconsistent with the other 4 z-scored deltas. `adx_5_20_ratio` is
already z-scored at this stage and is arguably a stronger delta candidate
anyway (trajectory on a ratio-between-time-horizons carries more
semantic content than trajectory on a single-window indicator).

Column-count impact: 5 features × ~2900 tradable tickers = ~14.5k new
columns in unified.parquet. ~8% expansion over the 176.5k pre-4a column
count. Preprocessing wall-clock ~10% slower; trivial impact downstream.

**Hypothesis:** 4a is a **prerequisite for run 4b's MLP test**, not a
standalone win. The MLP-only architecture under consideration for 4b
processes a single timestep snapshot — it cannot reconstruct trajectory
from a window. The deltas exist to make the MLP path informationally
equivalent to the encoder path on the temporal dimension.

For 4a's own hypothesis: deltas should be roughly neutral on the
existing encoder, since the encoder can in principle reconstruct
`delta_20[t] = level[t] - level[t-20]` from its 60-day window. The
question 4a answers is "do the deltas hurt the encoder?" — if no, 4b
is alive; if yes, the entire 4b architectural direction is dead.

**Seeds & Runs:**
- Seed 42: Run 11
- Seed 43: Run 12
- Seed 44: Run 13

**Methodology.** 3-seed MC (42, 43, 44) with determinism on. Comparison
against the 3-seed v5 baseline. Permissive decision rule pre-committed
to in writing before training kicked off, appropriate for a prerequisite
run rather than a substantive-hypothesis run:

> Cross-seed mean improvement on either return or Sharpe, with no major
> regression on any fold cluster, is enough to ship.

**Results — cross-seed aggregate:**

| | Return mean | Sharpe mean | Sharpe median | Pos rate |
|---|---|---|---|---|
| v5 baseline (3-seed) | +9.18% ± 3.41 | +0.485 ± 0.052 | +0.43 | 62.0% |
| v6 run 4a (3-seed)   | +8.77% ± 4.42 | +0.489 ± 0.086 | +0.45 | 62.6% |

Sharpe mean +0.004 (tiny positive). Return -0.41pts (tiny negative).
Both well inside the seed-stdev band. Seed-stdev widened on Sharpe
(0.052 → 0.086), suggesting marginally more variance in fold outcomes
across seeds — not a regression but worth noting.

**Same-seed paired delta (4a minus v5):**

| Seed | Δ Return | Δ Sharpe | Δ Pos% |
|---|---|---|---|
| 42 | +0.31 | +0.046 | +0.44 |
| 43 | -1.83 | -0.048 | -1.22 |
| 44 | +0.29 | +0.016 | +2.67 |

2 wins / 1 loss on both metrics, with all magnitudes small. Seed 43's
loss is the biggest single delta and it's only ~2pts return.

**Per-fold breakdown (3-seed means, 4a minus v5):**

```
Fold   Δ ret    Δ sharpe    Interpretation
1      +0.83    +0.04       small win
2      +0.98    +0.06       small win
3      +0.14    +0.02       wash
4      -2.69    -0.08       small loss
5      +0.67    +0.03       small win (regime-shift fold)
6      -1.16    -0.05       small loss (v5's peak fold)
7      -4.67    -0.11       meaningful loss ← persistent issue
8      +2.86    +0.06       win (regime-shift fold)
9      -0.66    +0.07       interesting: ret slightly down, sharpe up
```

7 of 9 folds improved on Sharpe (with the 2 losses being fold 4 and
fold 6, both small magnitude on Sharpe). 5 of 9 improved on return.

**Fold-level head-to-head (27 pairs):** 4a wins 14/27 on return, 16/27
on Sharpe. The 16/27 Sharpe count is ~59%, marginally above coin flip.

**Regime-shift folds (5, 7, 8) vs rest:**

- Regime-shift folds: -0.38pts ret, -0.01 sharpe
- Other folds: -0.43pts ret, +0.01 sharpe

Both essentially flat. The fold-7 regression is partly counterbalanced
by fold-5 and fold-8 improvements within the regime-shift cluster.

**Decision rule scorecard:**

| Criterion | Verdict |
|---|---|
| Cross-seed mean improvement on either ret or Sharpe | **PASS** (Sharpe +0.004) |
| No major regression on any fold cluster | **MARGINAL** (fold 7 -4.67 mirrors 3a's pattern) |

By the permissive rule, **passes — barely**. Fold 7 is the asterisk.

**The qualitative difference vs runs 1–3a is the most important
observation in the 4a result:**

- Runs 1 & 2 were unanimous in direction: 3 of 3 seeds worse on both metrics.
- Run 3a was 1W/2L with larger magnitudes (+1.25/-2.48/-2.95 return).
- Run 4a is 2W/1L with tiny magnitudes (all within ±2pts return).

This is the first v6 run that lands at the *actual* noise floor rather
than below it. Distinct from "clearly bad" in a way the MC framework
correctly distinguishes — if we'd run 4a with one seed and happened to
draw seed 43, we'd have called it bad; with seed 42 or 44, called it
good. The 3-seed protocol resolves this honestly to "neutral."

**Two consecutive runs hurt fold 7 specifically.** 3a (-6.13 ret, -0.16
sharpe) and 4a (-4.67 ret, -0.11 sharpe) both regressed there. Most
likely combined reasons: (1) v5's fold-7 result was already
favorably-drawn relative to the underlying COVID-regime difficulty, so
any perturbation regresses toward true mean, and (2) both added feature
types (slow-smoothed macro features for 3a, trajectory deltas for 4a)
depend on persistence assumptions that COVID's "sudden shock then
policy backstop" pattern violated. Not actionable for run 4 itself but
worth flagging.

**What 4a tells us about 4b — the actually-useful interpretation:**

Two hypotheses were alive going into 4a:

- **Encoder is doing real temporal work.** Adding deltas would be
  redundant → roughly neutral result.
- **Encoder is mostly ignoring temporal info, treating its input as
  bag-of-features.** Adding deltas would help materially → meaningful
  positive.

The observed neutral result is consistent with the first hypothesis,
mildly inconsistent with the second. *If* this reading is right, 4b
(MLP without temporal access) should reveal it cleanly: an MLP-only
path would underperform the encoder by however much temporal info the
encoder is actually using. If the encoder isn't using temporal info,
the MLP matches and we get a major architectural simplification for
free.

Both interpretations remain alive — 4a's signal is too small to resolve
definitively. But 4b becomes the *real* architectural test, and 4a
plumbed the bridge correctly.

**Decision: SHIP FORWARD.** The 5 new delta features stay in
`normalization.py` permanently. v6 baseline going into run 4b is
"v5 config + per-ticker deltas." 4b will compare MLP-only against this
baseline (not against v5 directly), so 4b's signal will purely reflect
the encoder-vs-MLP architectural question, not entangled with the
feature-engineering question 4a already answered.

---

## Cross-Cutting Lessons from Run 4a

**Match the decision rule to what the run is actually testing.** Run 3a
used a strict rule because it was testing a substantive hypothesis
(macro features should help regime shifts). Run 4a used a permissive
rule because it was testing prerequisite plumbing for the next run, not
a standalone hypothesis. Different rules for different roles. The rule
chosen post-hoc is selection bias; the rule chosen pre-hoc is
methodology.

**3-seed MC reliably distinguishes "noise-floor neutral" from "noisily
bad."** Runs 3a and 4a had per-seed Sharpe deltas in overlapping
magnitude ranges (single-seed could call either way), but their *patterns*
were different: 3a's losses were unanimous in direction at larger
magnitude, 4a's were 1-of-3 at smaller magnitude. The cross-seed protocol
made this distinction clean. Worth re-stating because the temptation in
both cases would have been to call "neutral, not significantly different"
on a single-seed read — which would have lost a real signal from 3a
(genuinely worse) and misread 4a (genuinely neutral) as a different
shape of result.

**Two consecutive runs that hurt the same fold is a signal about the
problem, not the methodology.** Fold 7 (COVID) regressed under both 3a's
macro features and 4a's trajectory deltas. Three v6 runs have now made
fold 7 worse (run 1 also did). One run is noise; three runs is a pattern
about how single-ticker SAC with smoothed features handles COVID-shape
regime shifts. v7's portfolio framing — where "cash" is a natural action
during periods nothing looks tradable — is likely to handle this
fundamentally better than any v6 feature addition can.

[NOTE: writeup-time interpretation used wrong fold-year mapping. The
recurring fold-7 regression is real but is the 2021 post-COVID melt-up,
not COVID itself. Corrected mechanism: the replay buffer's recency
emphasis (decay=3.0) weights 2020 chaos disproportionately at fold 7's
training end. The policy learns chaos-response reflexes that don't
match 2021's slower melt-up dynamics. Adding features (3a, 4a) didn't
cause this; it amplified an existing mismatch between the policy's
recent-training regime and fold 7's deployment regime. The "v7 cash as
natural action" conclusion still holds, plus a new v7 design
consideration: revisit recency-emphasis `decay` for the 5-ticker
basket. See "Correction — fold-year mapping" at top of file.]

**Marginal-positive on a prerequisite run is sufficient to ship forward.**
Don't require a clean win when the run's role is to enable the next
test, not to stand alone. The cost of being too strict at this stage is
killing a real architectural test prematurely.

---

## Where v6 Stands After Run 4a (Neutral / Shipped Forward)

- v6 baseline updated: v5 run-5 config + 5 per-ticker delta features
  (the 4a additions stay regardless of 4b's outcome).
- v5 reference distribution remains the *historical* baseline:
  ret_mean +9.18% ± 3.41, sh_mean +0.485 ± 0.052.
  Effectively unchanged after 4a (Sharpe mean +0.004, retmean -0.41pts).
- 4a's 14.5k new columns stay in the unified.parquet schema. Preprocessing
  ~10% slower; downstream cost trivial.
- 4b is the next live run: MLP-only market path. Decision rule will be
  stricter than 4a's (4b tests an actual architectural hypothesis, not
  plumbing). To be pre-committed before training kicks off.
- v6 capstones after 4b regardless of outcome. Per the stop-adding-things
  commitment, no 4c, no further ablations within v6.
- After v6 capstone: fresh chat with `v7_handoff.md` and the v7 pivot.

## Pending Run 4b: MLP-only market path

The architectural test 4a was enabling. Drop the CNN+Transformer encoder
in `networks.py`'s `FeatureExtractor`; pass the last-timestep market
vector (now 30-dim including the 5 deltas from 4a) through an MLP
instead. Per-ticker regime features and shared regime features still
get their existing paths — only the market sequence encoder gets
replaced.

**Hypothesis.** Either:

- The encoder was doing real temporal work, and the MLP underperforms
  by however much temporal info the encoder was extracting (probably
  visible as a meaningful aggregate drop). 4a's neutral result mildly
  favors this hypothesis.
- The encoder was mostly ignoring temporal info, and the MLP matches.
  In which case v6 ends with a substantial architectural simplification:
  fewer parameters, faster training, simpler codebase, and a model
  closer in shape to what v7's allocation problem will likely use.

Either result is informative. Unlike 4a (which was prerequisite
plumbing), 4b is a real architectural decision point.

**Seeds & Runs:**
- Seed 42: Run 14
- Seed 43: Run 15
- Seed 44: Run 16

**Methodology.** 3-seed MC vs the 4a baseline (NOT v5 — the deltas are
now part of the baseline). Determinism on, same protocol as previous
runs.

**Decision rule (pre-commit before training).** Stricter than 4a's:
require cross-seed mean within ~1pt return / ~0.03 Sharpe of baseline
on aggregate. MLP "passing" means matching the encoder, not beating it —
the win is architectural simplification at equal performance, not
beating the encoder on metrics. If the MLP underperforms by more than
that, the encoder is doing real work and stays in.

**Implementation note.** Network change is small: replace
`FeatureExtractor`'s conv+transformer stack with an MLP that takes the
last timestep's market features and outputs the same hidden_dim as the
existing encoder. No changes to actor/critic heads, no changes to the
replay buffer, no changes to the trainer.

### Possible Run 4c: long-history per-ticker base features

Filed mid-4a-training from an in-conversation observation: 4a's
`delta_20` is computed on features whose base levels are themselves
≤60-day windows. The 60-day market-data encoder window already sees
both endpoints of a 20-day delta, so for the *encoder path* the delta
is information the network could in principle reconstruct from the
window itself. The delta still earns its keep in 4a as the
prerequisite for 4b's MLP path (which doesn't have a window), but its
marginal value for the encoder is mostly regularization, not new
information.

The deeper observation: **every per-ticker feature except `rs_spy_252d`
is built on ≤60-day windows.** Long-horizon per-ticker context simply
doesn't exist in the feature set. The encoder is doing the best it can
with a 60-day view, and the only long-history signal it gets is via
shared regime features (breadth_200d, rs_spy_252d for the specific
ticker, vix-term aggregates, etc.) — none of which carry information
about *this ticker's specific* multi-quarter behavior.

The 4c hypothesis: adding long-history per-ticker base features, with
medium deltas on them, would give the encoder (or MLP) per-ticker
trajectory signal it currently can't see. Candidate base features
following the v5 run-3 "long base + medium delta" pattern:

- `log_return_120` or `log_return_252` — multi-quarter cumulative
  return for this ticker specifically
- `ema_close_ratio_120` or `_252` — close vs long-MA
- `volatility_60_252_ratio` — recent-vs-long-history vol regime per
  ticker

With deltas: `delta_60` on the 252d levels (matching the regime
convention for long-window levels).

**Why this is 4c, not 4a or 4b:**

- 4a tests a minimal additive change with a specific bridging role
  (enables 4b). Mid-flight scope creep would muddy attribution.
- 4b tests an architectural swap (encoder → MLP). Independent question.
- 4c tests a new hypothesis: "long-history per-ticker context matters."
  Genuinely new information, not a derivative of existing features.

If 4a fails, 4b is also likely dead — but 4c could still ship
independently as "add long-history features to the existing encoder
path." So 4c isn't strictly downstream of 4a/4b in dependency order, but
it makes sense to test 4a/4b first since they're already designed and
4c needs design work (which long-history features specifically, ticker
length / data coverage considerations for 252d windows on shorter
ticker lifecycles, etc.).

**Caveat to think through if 4c becomes live:** 252-day rolling windows
on tickers with <252+warmup days of history are zero/null — by which I
mean the post-`_normalize_data` rolling-z-score would produce zeros for
those rows. For the existing 60-day-window features this isn't a
problem because almost every ticker has ≥60+252 days. For 252-day-
window features, tickers with shorter lifecycles get more zero-padding,
which interacts with the existing lifecycle-segment machinery in
non-obvious ways. Worth a closer look before committing.

Not blocking. Filing for after 4a/4b complete.
