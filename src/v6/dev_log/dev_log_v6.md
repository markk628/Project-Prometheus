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

**Versions and Runs:**
- v5: runs 6, 7, 8 (seeds 42, 43, 44)
- v6: runs 2, 3, 4 (seeds 42, 43, 44)

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
- 6 runs total

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

**Versions and Runs:**
- v5: runs 6, 7, 8 (seeds 42, 43, 44)
- v6: runs 5, 6, 7 (seeds 42, 43, 44)

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

## Where v6 Stands After Run 2 (Rejected)

- v6 baseline = v5 run-5 config exactly. Both Sharpe and win_rate restored
  to portfolio_state (7 dims). Same as it was before run 1.
- Determinism hardening stays in for all future runs.
- 3-seed MC remains the methodology for every v6 ablation.
- v5 reference distribution: ret_mean +9.18% ± 3.41, sh_mean +0.48 ± 0.05.
  Bar to beat for the rest of v6.
- Subtractive portfolio_state work closed. Additive work deferred to v7+
  per `v6_handoff.md` v7 section.
- v6_handoff run sequence: with both portfolio_state runs rejected,
  the next live work is what was originally "Run 2" in the v5-era handoff
  — new regime features. That's now v6 run 3.

## Pending Run 3: More Regime Features

Plan unchanged from v6_handoff. Sector rotation (XL* ETFs), yield curve
(TLT/SHY or TLT/IEF ratio), credit spread (HYG/LQD), dollar regime (UUP),
commodities (GLD, USO), international RS (EFA, EEM, EWJ vs SPY), size
factor (IWM/SPY, MDY/SPY), growth factor (QQQ/SPY).

Higher EV than another long-horizon-of-same-thing addition, and run 1's
fold-7 result specifically points at "macro regime context during regime
shifts" as a weak spot — adding macro regime features is exactly what
addresses that. Run 2's fold-7 hit reinforces this.

3-seed MC, same standard.