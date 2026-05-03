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

## Where v6 Stands After Run 1 (Rejected)

- v6 baseline = v5 run-5 config exactly. Sharpe back in portfolio_state
  (7 dims).
- Determinism hardening is in and stays in for all future runs.
- 3-seed MC is now the methodology for every v6 ablation.
- v5 reference distribution saved for cross-run comparisons.
- v6_handoff run sequence shifts up one: the "remove Sharpe from
  portfolio_state" run is done and rejected; original run 2 (new regime
  features) becomes run 1.

## Pending Run 2 (originally Run 2 in handoff): More Regime Features

Plan unchanged from v6_handoff. Sector rotation (XL* ETFs), yield curve
(TLT/SHY or TLT/IEF ratio), credit spread (HYG/LQD), dollar regime (UUP),
commodities (GLD, USO), international RS (EFA, EEM, EWJ vs SPY), size
factor (IWM/SPY, MDY/SPY), growth factor (QQQ/SPY).

Higher EV than another long-horizon-of-same-thing addition, and run 1's
fold-7 result specifically points at "macro regime context during regime
shifts" as a weak spot — adding macro regime features is exactly what
addresses that.

3-seed MC, same as run 1.
