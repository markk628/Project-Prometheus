# v7 Dev Log

Working notes from the v7 daily-bar SAC project. v7 pivots the project from
single-ticker timing (the framing v5 and v6 shared) to allocation across a
small fixed basket of asset-class ETFs.

For prior history see `dev_log_v5.md` and `dev_log_v6.md`. For the original
v7 plan see `v7_handoff.md`. For preprocessing see `preprocessing.md`.

---

## v7 in one sentence

Stop predicting individual stock returns. Pivot to allocation across a
5-ticker basket of asset-class ETFs.

## Why this is the right pivot

v6 capstoned at +10.47% return / +0.484 Sharpe (3-seed MC) — roughly the
realistic ceiling for single-ticker SAC timing at this data scale, confirmed
across six v6 ablations. The remaining alpha lives upstream of "which
specific stock to time": systematic equity strategies in practice are mostly
asset-class allocation, and the allocation problem has structurally higher
signal-to-noise. See `v7_handoff.md` for the full argument.

---

## Pre-run setup

### Basket committed: asset-class diversity

```
SPY  — broad market equities
TLT  — long-duration treasuries
GLD  — gold
USO  — oil
UUP  — dollar
```

Default lean from `v7_handoff.md`. Picked because (a) it gives the cleanest
macro diversification story across five major asset classes, (b) the existing
regime feature library (breadth, VIX term structure, RS-vs-SPY, yield curve,
credit spread, size/growth factor, sector rotation) maps directly to
allocation decisions across these instruments, and (c) all five have
full-timeline data, so no universe-survivorship adjustment is needed.

The two other candidate baskets from the handoff (sector slice, risk-on/off)
are filed under Pending Work for later sweeps.

One known quirk worth flagging: with SPY in the basket AND used as the
RS-vs-SPY baseline, the `SPY_rs_spy_*d` columns are trivially zero for all
rows (SPY return minus SPY return). The columns exist to keep the per-ticker
feature schema uniform across basket members; the network sees them as a
constant-zero input and either ignores them or learns that there's no
predictive signal there. Goes away automatically if we ever switch to a
basket that doesn't include SPY — the sector-slice basket doesn't, so
that quirk is asset-class-basket-specific.

### v7 starting-state preprocessing decisions

These changes were made during the v6 → v7 preprocessing migration and are
live in the v7 starting state. Documenting them here so the baseline shape
is unambiguous when we look back.

- **Long-horizon regime features added.** `breadth_200d` (new long-horizon
  breadth level) with `_delta_20` and `_delta_60`. RS-vs-SPY long-horizon
  (`rs_spy_252d` + 20d/60d deltas) was already in v6 capstone — unchanged.
  VIX term structure deliberately stays at 5/20/60d only; it's event-driven
  by design and a 200d smooth would destroy the transitions that matter.
  Rationale: the "long base + medium delta" pattern that worked for
  per-ticker in v5 run-3 and v6 run 4c, now applied to regime signals.

- **Macro regime features active.** The 5 macro families from v6 run 3a
  (yield curve, credit spread, size factor, growth factor, sector rotation
  — 18 columns total) are wired into `_build_unified` in v7. Strictly they
  were supposed to be reverted after v6 rejected 3a, but the revert never
  landed. Keeping them for v7 because the rejection rationale ("broad-spectrum
  macro context doesn't transfer to single-ticker timing") doesn't apply —
  v7 IS the allocation problem 3a's macro features were a better fit for.
  The v7_handoff anticipated revisiting them as a possible run 6+; in
  practice they're now baseline, which means "do macros help v7" becomes
  a subtractive ablation (see Pending Work) rather than the additive run
  originally planned.

- **Wiring through `SHARED_REGIME_PREFIXES`.** The trainer's
  `load_tickers_from_unified` uses the full union (breadth + vix + macro)
  for regime-column classification, so every shared regime column above
  feeds the network's regime channel via `Actor.regime_fc` and
  `Critic.q*_regime_fc`.

### Structural changes needed before run 1

The v7 pivot from single-ticker timing to 5-ticker allocation requires
changes to environment, networks, agent, and trainer that have NOT yet
landed. The v6 capstone trainer/env/agent are still in place in
`src/v7/model/`. Work:

- **Break the "regime = context only, tradable = trained on" binary.**
  Basket members are regime tickers (SPY/TLT/GLD/USO/UUP all live in
  `REGIME_TICKERS` today). They need to be BOTH: still feed the
  shared-regime computations AND have their per-ticker columns retained
  in `unified.parquet`, with the trainer admitting them as tradable.
  Concrete touches:
    - `feature_engineer.py::_build_unified` drops `_close` columns for
      regime tickers before write (except `SPY_close` as benchmark) —
      basket members need an exemption.
    - `trainer.py::main` filters with `not in regime_set` when discovering
      tradable tickers — needs to admit basket members.
    - `_cross_sectional_normalization` excludes regime tickers from the
      CS population. Leave as-is for the asset-class basket (TLT vs ~3000
      stocks isn't a meaningful CS comparison) — basket members will be
      shaped as per-ticker market features + RS-vs-SPY only, no
      `_cs_zscore` / `_sector_zscore` features. Sector one-hots also
      absent (all 5 basket members map to sector 11 "other" via
      `ETF_SECTOR_OVERRIDES`, so per-ticker one-hots would be 100%
      identical across the basket — useless within-basket signal).
      Confirmed shape post-preprocessing: 43 features + 1 close per
      basket member.

- **Environment rebuild.** `DailyEnvironment` is single-ticker, action
  space `Box(shape=(1,))`, `PORTFOLIO_STATE_DIM=7`. v7 needs:
    - 5-dim continuous action space (target allocation per ticker, cash
      as implicit residual; weights sum to ≤ 1).
    - Multi-asset portfolio state tracking (per-ticker positions, P&L,
      hold times, total exposure).
    - Reward function from joint portfolio return, not single-ticker
      return.
    - Decision on rebalancing constraint vs turnover-cost-in-reward.
    - Decision on initial allocation (equal-weight vs all-cash — handoff
      leans all-cash for the "active value-add over passive baseline"
      framing).

- **Networks update.** Actor outputs 5-dim, Critic takes 5-dim action.
  Encoder topology question (shared encoder across the 5 tickers vs 5
  parallel encoders) needs to land. Start lean: shared encoder; only
  test parallel if a specific signal points there.

- **Trainer / episode structure.** Episodes change from "single ticker
  over 252 days" to "5-ticker portfolio over 252 days." Need a new
  episode-sampling scheme — overlapping start-date windows on the basket,
  vs the v6 ticker-sampling approach that no longer applies. Sample count
  per fold will drop substantially (no cross-ticker sampling diversity)
  so the right replacement structure for that diversity needs thought.

- **Replay buffer compatibility.** `DailyReplayBuffer` flat-state width
  depends on portfolio-state size and per-ticker market dims; needs
  recomputation once portfolio state shape is finalized.

### What carries forward unchanged

- **Walk-forward fold structure.** 9 folds, valid years 2015-2023,
  2024-2025 held out as test (`data_start_year=2005`, `initial_train=10`,
  `valid=1`, `test=2`).
- **3-seed MC discipline.** Seeds 42/43/44 per condition for every
  ablation.
- **Determinism setup.** Full RNG coverage + `cudnn.deterministic=True`
  + `cudnn.benchmark=False`.
- **Per-fold MC methodology.** Single-variable changes per run,
  pre-committed decision rules in writing before training kicks off,
  document rejections as cleanly as acceptances.
- **MLP-only encoder architecture (v6 run 4b).** 3-layer MLP on the
  last-timestep snapshot, no CNN/Transformer. v6's 4b answered the
  encoder-vs-MLP question for single-ticker; same answer probably ports
  to allocation. Revisit only with a specific reason.

### Reference distributions (for context, not comparison)

```
v5 (3-seed MC, single-ticker):  ret +9.18%  ± 3.41   sh +0.485 ± 0.052
v6 capstone (3-seed MC, single-ticker):  ret +10.47% ± 3.91   sh +0.484 ± 0.092
```

**Not directly comparable to v7.** v6's universe inherited survivorship
bias from full-timeline filtering before walk-forward split. v7's
hand-picked basket sidesteps the universe-selection problem entirely
but is solving a different problem on a different universe. v7's
reference point is buy-and-hold of the basket (or SPY-only as a
single-asset reference), not v5/v6 Sharpe.

---

## Pending work

Items deferred or filed for later. Add to / strike from as v7 progresses.

### Alternate basket compositions

Two other baskets to sweep once asset-class diversity has a v7 baseline:

- **Sector slice.** XLK, XLF, XLE, XLV, XLY — pure sector rotation across
  major economic sectors. Different problem shape: less asset-class
  diversification, more cross-sectoral timing within equities. Tests
  whether the allocation framing generalizes from cross-asset rotation
  to within-asset-class sector rotation.
- **Risk-on/risk-off.** SPY, QQQ, GLD, TLT, IWM — captures the main
  allocations a real systematic equity strategy would shift between.
  Most equity-tilted of the three; closer in spirit to v6's "should I
  be in stocks right now" framing but with diversification across the
  equity leg.

### Recency-emphasis sweep

The single most important diagnostic finding from v6: `IndexReplayBuffer`
with `decay=3.0` is upstream of the persistent fold-7 (2021 melt-up)
regression. Five qualitatively different v6 runs all hurt fold 7
disproportionately; the v6 dev log diagnosed this as the recent third of
the buffer weighting 2020 chaos disproportionately at fold 7's training end.

v7 uses `DailyReplayBuffer` (different buffer for multi-ticker training)
but with the same `decay=3.0` default in `trainer.py::main`. Sweep is
high-EV before locking v7's baseline — but should run on the v7 basket
setup, not on v6 single-ticker, because the buffer dynamics differ (no
cross-ticker mixing in a fixed 5-ticker basket, smaller absolute buffer,
much higher per-ticker transition density).

Plan: after the basket/env/networks land but before run 1's full MC, sweep
`decay ∈ {0.0 uniform, 1.0 mild, 3.0 v6 default}` on a single fold with a
single seed, lock the choice, then run 1's MC fires against that locked
baseline.

### Macro feature subtractive ablation

Since the 5 macro regime feature families are now active in the v7
starting baseline (see "v7 starting-state preprocessing decisions"
above), testing "do macros help v7" becomes a subtractive ablation
rather than the additive run 6+ originally planned in the handoff:

- Baseline: v7 run-N with macros on (current state)
- Ablation: baseline minus the 5 macro families, 3-seed MC
- Compare aggregate + per-fold

v6's "subtractive ablations closed" rule was specifically about
portfolio_state dims (Sharpe / win_rate), not all feature families. A
macro-subtractive test in v7 is still on the table when it's worth running.

### Portfolio-state deltas

v6 runs 4a (medium-horizon per-ticker deltas) and 4c (long-horizon
per-ticker deltas) were two of the highest-yield additions of the v6 cycle,
suggesting delta-of-z-score is a broadly useful pattern. The same argument
applies to portfolio state in v7: the run-1 baseline ships a 22-dim
portfolio snapshot, but portfolio state is inherently path-dependent
(scaling in/out, P&L acceleration, rebalancing drift) and a snapshot drops
all of that information.

Candidate dim additions per delta horizon W:

- Per-ticker (×5): `current_weight_delta_W`, `position_return_delta_W`,
  `distance_from_target_delta_W`. (`hold_time` excluded — monotonic by
  definition, delta is trivially constant.)
- Portfolio-level: `cash_fraction_delta_W`, `total_value_delta_W`.

At one horizon W (e.g. 5d) that's +(3×5 + 2) = +17 dims, taking portfolio
state 22 → 39. Two horizons (5d + 20d) → 56. Implementation cost: env
maintains a small history buffer to compute deltas at step time.

Sequencing within v7: probably belongs as an early addition alongside or
before the portfolio-state window run on the handoff's indicative sequence.
The argument for deltas-before-window is the same as v6 4c — deltas
pre-encode the temporal question at lower bandwidth than a full window,
and the v6 evidence is that deltas often deliver the expected gain without
needing the full window's capacity.

### Target-entropy ablation (sustained exploration as regularizer)

**Note: this section was rewritten mid-run-1 after seeing the full SAC
metric curves. The original version misdiagnosed target_entropy=-5 as
structurally unreachable; the actual mechanism and proposed ablation
shift accordingly.**

Run 1 trajectory: alpha starts at 1.0, decays monotonically through
folds 1-5, equilibrates at ~0.01 around ep 1000 (start of fold 6).
Entropy starts at +3.4, holds roughly constant through folds 1-3, then
crashes from +3 down to -5 between ep 600 and ep 1000, stabilizes at -5
from fold 6 onwards. Alpha_loss tracks this: monotonically negative
(pushing alpha down) through folds 1-5, oscillates around 0 from fold 6
onwards. Equilibrium reached cleanly.

**Corrected mechanism: `target_entropy = -|A| = -5` IS reachable,**
despite the `log_std_max = 0` clamp. While the Gaussian portion of
log_prob is bounded by std=1, the tanh correction term
`-log(1 - tanh²(x) + ε)` grows unboundedly negative as actions saturate
near ±1. As alpha decays and the policy concentrates its mean further
from 0, sampled actions push toward the tanh boundaries, the correction
term dominates, and entropy dives into negative territory. The system
equilibrates at entropy ≈ -5 with alpha at a small positive value
(~0.01) — exactly the SAC auto-tuning equilibrium by design. The
original "two compounding reasons it can't reach -5" diagnosis was
wrong because it failed to account for the unboundedness of the tanh
correction.

By fold 6 the SAC mechanism has fully converged: critic loss ~0,
Q-value stable at ~25, actor loss flat at ~-25, alpha at floor, entropy
at target. From fold 6 through fold 9 there is no meaningful internal
dynamics change — the policy is in a "converged-and-overfit" steady
state where the critic perfectly fits the deterministic policy's narrow
transition distribution. The "still learning at fold 9" pattern from v6
(critic loss declining, Q rising — what motivated UTD bumping there)
does NOT apply to v7. Bumping UTD here would do nothing because the
critic loss is already essentially zero.

Whether this convergence-to-determinism is good or bad depends on
whether full convergence is what we want for an allocation strategy.
Run 1's cross-fold validation suggests not — the converged policy
memorizes its training regime and crashes on the next year's regime
shift (40-100pp drops at fold transitions). Sustained exploration via
auto-tuning could function as a regularizer.

Proposed ablation: set `target_entropy` to a value the policy CANNOT
reach (or can only reach with very wide outputs), forcing alpha to
stay positive and the policy to stay stochastic. Three candidate values:

- **`target_entropy = -5` (current baseline):** policy reaches target by
  going fully concentrated, alpha → ~0.01, exploration effectively
  shuts off by fold 6.
- **`target_entropy = +1`:** requires moderate sustained entropy. Alpha
  would stabilize at a moderate positive value, exploration continues.
- **`target_entropy = +3`:** forces high entropy. Alpha stays elevated,
  policy can't fully exploit even if it wants to. Strongest exploration
  regularizer.

Hypothesis-driven selection: if the issue is "policy locks in too
hard," +1 is the conservative test. If it's "policy needs continuous
exploration to avoid regime memorization," +3 is more aggressive.
Probably worth testing both against the -5 baseline so the relationship
between target value and generalization is visible.

Sequencing within v7: target_entropy is one regularizer among several
the run 1 results indicate we need (L2 weight decay, dropout, early
stopping). Cleanest experimental structure is run 2 with the
traditional regularizer stack (L2 + dropout + early stopping) as the
first regularization test, then layering target_entropy adjustments
in run 3 if cross-fold drops persist. Avoids confounding "which
regularizer helped" if everything is changed at once.

### Other items from v7_handoff

The v7_handoff lays out an indicative run sequence (run 1 baseline →
run 2 per-ticker features in observation → run 3 portfolio-state window
→ run 4 cumulative trajectory dims → run 5 turnover penalty → possible
6+). Not duplicated here. As runs land, each gets its own writeup
in this dev log under standard `## Run N — [name] ([VERDICT])` headers.

---
