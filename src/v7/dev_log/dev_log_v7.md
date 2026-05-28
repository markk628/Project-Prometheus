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

## Runs

### Run-1 NaN incident (pre-baseline, debugging)

Before run 1 produced usable results, training crashed three times at
fold 1 ep 55-56 with NaN actor outputs (`Normal(loc=nan)`). Root cause
chain:

1. **5-dim summed log_prob overflow.** SAC's log_prob sums the per-dim
   tanh correction `-log(1 - tanh²(x) + ε)` across 5 action dims. With
   ε=1e-6, each dim caps at ~-13.8, summed -69 worst-case. Fed through
   the SAC target `Q = r + γ(Q_next - α·log_prob)`, this produced target
   magnitudes that overflowed float32 during MSE squaring → NaN
   gradients. The cascade started as alpha decayed (ep 55, alpha ~0.76)
   and the policy began concentrating, pushing actions toward tanh
   saturation.
2. First fix attempt (gradient finiteness check before optimizer.step)
   was insufficient — params still went NaN via Adam internals even
   when grads were finite at the check.

Final fixes (all retained in v7 baseline):
- tanh-correction ε bumped 1e-6 → 1e-4 (caps per-dim correction at -9.2,
  sum -46, inside float32 safe range)
- `log_std_max` tightened 2.0 → 0.0 (max per-dim std 7.4 → 1.0; the v6
  setting was fine for 1-dim action but produced absurdly wide joint
  exploration in 5-dim)
- two-stage NaN guard in agent: pre-step grad-finiteness check AND
  post-step param-finiteness check with rollback to a pre-step snapshot;
  skip counters surfaced in logs
- alpha optimizer given the same gradient clipping as actor/critic
  (was inconsistently missing)

Also fixed during this period: a pre-IPO NaN-price bug (USO IPO
2006-04, UUP IPO 2007-02, both post-CUTOFF 2004-12) where episodes
sampling start_idx before a basket member's IPO hit NaN prices.
`_build_basket_inputs` now computes the basket-wide valid index range;
trainer clamps sampling to it. Binding constraint is UUP's 2007-02-20
IPO. Note: the original dev-log claim that "all five basket members
have full-timeline data" was wrong — fold 1's effective training
window is ~2 years shorter than later folds because of this.

### Run 1 — baseline (no regularization)

9 folds × 200 episodes, walk-forward, validation = year N+1.

**Aggregate validation:** mean +7.81%, median +1.11%, STD 28.26%,
positive 102/180 (56.7%), Sharpe mean +0.42, Sharpe max 3.83.

**Per-fold best validation:**

| Fold | Valid yr | Best ret | Best Sharpe | Note |
|------|----------|----------|-------------|------|
| 1 | 2015 | -1.7% | -0.18 | all negative |
| 2 | 2016 | +3.5% | 0.41 | crossed zero |
| 3 | 2017 | +11.6% | 1.96 | strong climb |
| 4 | 2018 | +14.9% | 1.70 | recovered from regime drop |
| 5 | 2019 | -10.9% | -0.18 | CATASTROPHE: -27% in +29% SPY yr |
| 6 | 2020 | +120.5% | 3.83 | massive (overfit / COVID vol) |
| 7 | 2021 | +10.2% | 0.86 | -133pp drop at transition |
| 8 | 2022 | +3.7% | 0.14 | brutal rate-hike year |
| 9 | 2023 | +17.4% | 1.46 | final fold |

**Severe overfitting.** Train mean +100.1%, median +87.1%, Sharpe mean
4.92 — roughly 10x the validation numbers. Cross-fold drops of
40-133pp at every fold transition (same policy, new validation year).
The policy memorizes each training regime and crashes on the next.
Diagnosis: 5 fixed tickers means ~340k params vs ~12k unique trading
days per fold — capacity vastly exceeds data variety. v6 had ~3000
tickers as an implicit regularizer; v7 lacks it.

**SAC internals (from full metric curves).** Q peaks 207 (ep ~150),
collapses to ~25 by fold 4, stable thereafter (peak was alpha-bootstrap
inflation, collapse is alpha decay removing the entropy bonus from the
Q-target). Critic loss ~0 from fold 4 onward. Entropy crashes +3 → -5
between ep 600-1000, stable at -5. Alpha → ~0.01 by fold 6. **System
fully converged by fold 6** — no meaningful internal dynamics fold 6-9.
The v6 "still learning at fold 9, bump UTD" pattern does NOT apply;
critic loss is already zero. (This corrected an earlier misdiagnosis —
see Target-entropy section.)

### Run 2 — regularization (L2 + dropout + best-checkpoint tracking)

Same fold/episode structure as run 1. Changes: `weight_decay=1e-4` on
actor+critic Adam; `dropout=0.1` in Actor.trunk + Critic q1/q2 trunks;
Sharpe-based best-checkpoint tracking per fold; critic_target set to
eval() so dropout doesn't corrupt Bellman targets.

**Result: regularization did NOT help. Validation got slightly worse.**

| Metric | Run 1 | Run 2 |
|--------|-------|-------|
| Valid mean ret | +7.81% | **-1.18%** |
| Valid median ret | +1.11% | +0.50% |
| Valid >0 | 56.7% | 52.8% |
| Valid Sharpe mean | +0.42 | **-0.00** |
| Valid Sharpe max | 3.83 | 1.65 |
| Train mean ret | 100.1% | 17.2% |
| Train Sharpe mean | 4.92 | 0.75 |

The train-val gap shrank, but by **lowering the ceiling, not raising
the floor.** Train overfitting magnitude dropped sharply (train mean
100% → 17%, train Sharpe 4.9 → 0.75) but validation moved DOWN to meet
it rather than the reverse. Per-fold, run 2 was worse in the folds that
mattered: fold 6 peak 120% → 34%, fold 8 best +3.7% → -5.2% (never
positive), fold 9 peak +17.4% → +6.2%. Cross-fold cliffs unchanged
(fold 4→5 still -24%, fold 6→7 still -5.5%). SAC internal curves nearly
identical to run 1 (same Q peak/collapse, same entropy crash, same
alpha decay) — L2+dropout at these strengths was a small perturbation
on optimization, not a structural change.

**Interpretation: wrong class of fix.** We treated the overfitting as
excess-capacity-memorizing-noise, which capacity-constraint
regularization addresses. But the cross-fold cliffs are
**distribution shift between regimes** (2019 bull vs 2018 correction vs
2022 rate shock are different data-generating processes), not excess
capacity within a regime. L2/dropout shrink the model's ability to fit
*any* regime including the legitimate signal, without helping it
generalize *across* regimes. Can't regularize 2022 into being
predictable from 2008-2021 if 2022 is genuinely OOD.

**Per the pre-registered plan, run 2's null result is informative:** it
eliminates the capacity-constraint family. The L2-only and dropout-only
ablations are now unnecessary (they'd be weaker versions of a change
that already didn't move validation), saving two runs.

**Caveat:** these are single-seed runs. The val mean delta
(+7.8% → -1.2%) is probably beyond seed noise given the consistent
per-fold degradation and near-identical internal curves, but a clean
confirmation would need one more seed of each. Judgment call: trust the
signal and move on rather than spend compute confirming a negative.

**Next:** revert regularization (weight_decay → 0 or 1e-5, dropout
optional), move to target_entropy ablation (run 3) as the lever that
directly attacks the distribution-shift / regime-lock-in failure mode
rather than the capacity one. See Target-entropy section. Also
reconsider problem framing: cross-fold cliffs may indicate one fixed
policy can't serve all regimes (regime-conditioned policy, regime
embedding, or revised evaluation horizon are on the table).

### Revert + target_entropy plumbing (between run 2 and run 3a)

Reverted run-2 regularizers to inert defaults (kept as tunable knobs,
not deleted): `weight_decay` 1e-4 → 0.0, `dropout` 0.1 → 0.0 in
Agent/Actor/Critic (nn.Dropout(p=0.0) is a no-op, layers stay wired),
removed `critic_target.eval()` (only mattered with dropout active).
KEPT: best-checkpoint tracking (methodology), all NaN fixes, expanded
logging. Added a run-3 config block at top of `main()` with three
knobs (`target_entropy=None`, `weight_decay=0.0`, `dropout=0.0`) wired
through to the Agent, plus a startup log line `Agent config:
target_entropy=..., weight_decay=, dropout=` so each run self-documents.
**Baseline verification:** ran `target_entropy=None` → resolves to
-action_dim=-5; reproduced run 1 IDENTICALLY (confirmed clean revert,
so any run-3 delta is attributable to target_entropy alone).

### Run 3a — target_entropy = +1.0 (sustained exploration)

Single change vs run-1 baseline: SAC entropy target -5 → +1.

**Ablation engaged (confirmed via internals).** Unlike runs 1-2 where
entropy crashed to -5 and alpha decayed to ~0.01, run 3a entropy dips
to ~0.9 (ep 700) then holds at ~1.0 for the whole back half (mean
1.667, median 1.005) — the +1 target is binding. Alpha stabilizes at a
positive equilibrium ~0.05 (mean 0.137, median 0.054) instead of
collapsing. alpha_loss settles around 0 with two-sided noise (active
constraint) rather than one-sided decay. Critic loss now genuinely
near-zero from fold 4 (max 76.6 vs run-1 173.6) — less entropy-bonus
inflation of the Q-target. q_value peak similar (~207) but settles
lower (~28 vs run-1 ~45). This is a genuinely different optimization
trajectory, not a reseed.

**Aggregate validation vs prior runs:**

| Metric | Run 1 | Run 2 | Run 3a (te=+1) |
|--------|-------|-------|----------------|
| Valid mean | +7.81% | -1.18% | **+9.49%** |
| Valid median | +1.11% | +0.50% | **+3.30%** |
| Valid >0 | 56.7% | 52.8% | **67.8%** |
| Valid Sharpe mean | +0.42 | -0.00 | **+0.70** |
| Valid max | 33.9% | — | **117.1%** |

Both floor AND ceiling rose. Pre-registered prediction was half right:
correctly called the floor / positive-rate would improve, WRONG that
the ceiling would drop (it rose). Positive rate 56.7% → 67.8% is the
cleanest single number.

**BUT the failure mode is NOT fixed.** The cross-fold cliffs — the thing
target_entropy was supposed to attack — are still present and arguably
sharper. Validation is now a dramatic sawtooth: it climbs steeply
within each fold, then craters at every fold boundary. Fold 6 (2020)
climbs to +116.5% / Sharpe 4.17 by ep 1200, then fold 7 opens at
-3.99% — a ~120pp cliff, *bigger* than run-1's drop and from a much
higher peak. Fold 8 opens -7.74%, fold 9 opens +0.23%. The policy is
still completely regime-locked; exploration just lets it climb higher
within each regime (it doesn't commit prematurely, so it keeps
improving across a fold's episodes) before falling off when the regime
changes. The sawtooth IS the regime lock-in, at higher amplitude.

**Fold 6 caveat (important for interpreting the aggregate).** Fold 6
now reaches +116% / Sharpe 4.17 — even larger than run-1's 120%
question mark. Its late episodes inflate the aggregate mean
substantially; +9.49% is NOT evenly distributed (fold 6 and a strong
fold 9 carry much of it). The "is fold-6 genuine COVID-vol capture or
artifact" question, open since run 1, is now more load-bearing.

**Genuine within-regime improvements:** fold 5 (2019) recovers from
run-1's -27% catastrophe to +5.4%; fold 7 peaks +18.9% / Sharpe 1.76
(up from +10.2%); fold 9 +21.65% / Sharpe 2.26 (best fold-9 any run).

**Honest summary:** te=+1 raised within-regime performance and the
aggregate, but did NOT solve the cross-fold distribution-shift problem
it was designed to attack. Helpful, not curative.

**MC sweep now mandatory (not optional).** The whole run1→2→3a chain is
single-seed with no estimate of seed variance. The mechanism
demonstrably engaged, but whether +9.49% vs +7.81% is a reliable
ranking vs seed scatter is unanswerable from one seed each — especially
since fold 6 (noisiest fold) alone could swing the mean several points.
Plan: 3 seeds each of baseline (te=-5) and te=+1, same folds; report
mean ± std of valid-mean and valid-Sharpe AND per-fold spread (to test
whether fold-5 recovery and the fold-6 monster are stable or
seed-dependent — this finally answers the fold-6 genuineness question).
~6 run-days. te=+1 is "real" only if it beats baseline outside ±1 std
across seeds.

**Run 3b (te=+3) DEFERRED** until the sweep shows te=+1 is
distinguishable from baseline — no point testing a more aggressive dose
if the first one isn't measurably different from noise.

### Reporting-metric audit (post-run-3a) — Sharpe/Sortino + win-rate

Triggered by a run-3a fold-8 episode log showing Sharpe 7.13 alongside
Win Rate 20.00% with +308% return / PF 5.46 — which looked
contradictory but isn't. Findings (no code changed):

**These metrics are reporting-only — they do NOT touch training.**
Verified: `_get_info` is built every step and returned from `step()`,
but the per-step loop in `_run_episode` only pushes
`(state, action, reward, next_state, done)` to the replay buffer; it
never reads `info["sharpe_ratio"]` etc. Only `_log_episode_block`
(end of episode) consumes them. Agent/buffer never see `info`. Also
re-confirmed Sharpe/win-rate are NOT in the 22-dim portfolio state
(state = per-ticker {weight, position_return, hold_time_log,
dist_from_target} ×5 + {cash_fraction, total_value_log_ratio}); they
were in v6 state, removed in v7.

**Sharpe/Sortino are CORRECT as logged — no fix needed now.**
`_get_episode_sharpe` clips daily Sharpe to [-3,3] then `_get_info`
×√252 to annualize, so the effective reported ceiling is ~47.6
annualized. The cap has NEVER bitten in any run (train Sharpes ~10 →
daily 0.63; fold-6 monster 4.17 → daily 0.26 — all far under daily-3).
So every Sharpe/Sortino logged in runs 1/2/3a is the true value,
untouched by the clip. The only latent fragility is the `+1e-9`
denominator (a near-zero-vol episode would inflate the ratio, with the
cap currently the only backstop) — but no such episode has occurred.
**Decision: leave Sharpe/Sortino as-is.** They're correct, reporting-
only, and the clip is a harmless guard. The clipped-vs-annualized design
question only becomes live IF/when these get added to the portfolio
state (a state feature needs a bounded, daily, non-annualized version —
different quantity from the report metric). Defer that design to the
state experiment itself rather than pre-building it. Calmar scaling is
correct as-is (annual-return/maxDD, not √-time-scaled).

**Win-rate is misleading for this strategy class — slated for removal.**
`_get_win_rate` = winning_trades / completed_trades, where a "completed
trade" only counts when a ticker goes FULLY flat (shares < 1e-9). But
softmax weights have a ~0.05 floor (run-3a mins ~0.047) so positions
almost never fully close → win-rate is computed over a tiny subsample of
rare full-liquidations, not the continuous rebalancing that actually
generates returns. Worse, full closes happen when the policy EXITS a
name, which in a trending-up portfolio skews toward losers it's giving
up on (winners are held ~full episode, never counted). So low win-rate +
high PF is the EXPECTED signature, not a contradiction. Arithmetic is
correct; the metric is near-meaningless here. (PF 5.46 and avg win/loss
2.49 are fine — computed from the per-step return series, not closes.)
**Decision: remove win-rate from v7 logs (or redefine as positive-step-
return rate). Reporting-only, so removal is zero-risk — do it after the
MC sweep to avoid touching code mid-sweep.**

### Run 3a MC sweep — te=-5 vs te=+1, 3 seeds each (RESULT: te=+1 confirmed real)

3 seeds × {te=-5 baseline, te=+1}, 6 runs, same 9 folds, same code.
Resolves whether run-3a's te=+1 gain was above seed noise, and whether
the fold-6 monster is structural or luck.

**Seeds & Runs:**
- te=-5 (baseline): seed 42 = run 1, seed 43 = run 6, seed 44 = run 7
- te=+1:            seed 42 = run 3, seed 43 = run 4, seed 44 = run 5
- "run N" = the logged run label; all 6 were part of the run-3a sweep.
  (Note: te=-5 seed 42 IS the original single-seed "run 1" baseline —
  same seed, hence its +7.81% matches.)

**Aggregate (mean ± std over 3 seeds):**

| Metric | te=-5 | te=+1 | Separated? |
|--------|-------|-------|------------|
| Valid mean ret | +4.12 ± 4.66 | +7.11 ± 3.46 | partial overlap |
| Valid Sharpe | +0.257 ± 0.231 | +0.574 ± 0.171 | **clean (>1σ)** |
| Positive rate % | 55.9 ± 1.8 | 67.6 ± 1.4 | **clean, large** |

Per-seed valid means: te=-5 = [+7.81, -1.11, +5.67]; te=+1 = [+9.49,
+3.14, +8.70].

**Verdict: te=+1 is a genuine, replicated improvement — not seed noise.**
Clears the pre-registered bar (beat baseline by >±1σ) on Sharpe
(baseline +1σ=0.488 < te=+1 mean 0.574) and decisively on positive rate
(55.9±1.8 vs 67.6±1.4, no overlap — ~12pp more profitable evals, tight
across seeds). Raw mean return overlaps (baseline +1σ=8.78 vs te=+1
mean 7.11) — but mean return is the noisiest summary because fold 6's
huge variance blurs it; Sharpe and positive-rate (robust to one
fat-tailed fold) are the right statistics and both separate cleanly.

**KEY FINDING 1 — baseline is wildly seed-dependent; run-1's +7.81% was a
lucky seed.** Baseline valid-mean ranged +7.81 / -1.11 / +5.67 (9-point
swing); seed 43 baseline went NEGATIVE. The +7.81 the entire run-1
narrative was built on is the TOP of baseline's range. This
retroactively confirms that all earlier single-seed comparisons
(run1 vs run2, etc.) were reading deltas smaller than seed scatter and
were noise-dominated. Vindicates the decision to require the sweep.

**KEY FINDING 2 — fold 6's monster is SEED LUCK, not skill (long-open
question now answered).** Fold-6 (2020) best return per seed:
- te=-5: [120.5, 40.5, 86.3] → mean 82.5, std 40.1, CV 0.49
- te=+1: [117.1, 43.3, 96.2] → mean 85.5, std 38.0, CV 0.44

The 120% from run 1 was the high seed; same config seed 43 gives 40.5%
(3× swing from seed alone). CV ~0.5 ⇒ fold 6 is dominated by seed luck,
not structural skill. te=+1 and te=-5 are statistically
indistinguishable on fold 6 (82.5 vs 85.5, both ±~40) — so fold 6 is
NOT the source of te=+1's advantage. (It also explains the aggregate
mean-return overlap: fold-6 ±40 variance injects noise across the 180
evals.)

**Where te=+1's advantage actually comes from (per-fold best ret,
mean±std over seeds):**

| Fold | te=-5 | te=+1 | delta |
|------|-------|-------|-------|
| 1 (2015) | -1.70 ± 0.70 | -1.70 ± 0.70 | +0.00 |
| 2 (2016) | +3.56 ± 0.63 | +3.56 ± 0.63 | +0.00 |
| 3 (2017) | +8.47 ± 3.83 | +8.45 ± 3.74 | -0.01 |
| 4 (2018) | +8.70 ± 7.74 | +7.67 ± 5.19 | -1.03 |
| 5 (2019) | -10.50 ± 8.75 | -1.57 ± 10.02 | +8.93 |
| 6 (2020) | +82.45 ± 40.13 | +85.53 ± 38.02 | +3.08 |
| 7 (2021) | +14.56 ± 3.84 | +20.72 ± 3.44 | **+6.16** |
| 8 (2022) | +6.06 ± 2.10 | +16.31 ± 7.33 | **+10.25** |
| 9 (2023) | +10.98 ± 7.56 | +17.24 ± 4.52 | **+6.27** |

Folds 1-3 identical (policy hasn't diverged yet). Gains concentrated in
late folds. **Trustworthy improvements: folds 7, 8, 9** — consistent
sign, small-to-moderate std (7 and 9 especially tight). Fold 5's +8.93
is real in the mean but lives inside enormous variance (±~9-10
regardless of config) → "directionally better but unreliable." Fold 8
(2022 rate shock) is the largest reliable delta: +6.06 → +16.31.

**Standing conclusion:** te=+1 is a real but modest improvement
(~12pp positive rate, Sharpe 0.26→0.57), driven by steadier mid/late-
fold performance, NOT the fold-6 lottery. It did NOT fix the cross-fold
cliffs (run-3a finding stands) — it raised the within-fold level, the
sawtooth structure remains.

**Next decisions:**
- **Run 3b (te=+3) now JUSTIFIED** (the deferral condition — "te=+1
  distinguishable from baseline" — is met). Tests whether more
  exploration helps further or overshoots. Cost: 3 seeds = 3 run-days.
- **Regime-conditioning** remains the higher-leverage direction if the
  cliffs (still the dominant failure mode) are the real target —
  exploration only dented within-fold level, not the boundaries.
- Methodological lock-in: ALL future config comparisons must be
  multi-seed. Single-seed deltas in this system are noise-dominated
  (proven by the 9-point baseline swing).

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

Sequencing within v7: **run 3a (te=+1) DONE — see Runs section.** The
ablation engaged as designed (entropy held at ~1.0, alpha stabilized
positive ~0.05, confirming the +1 target binds — the predicted
mechanism). Outcome: it raised within-regime performance and the
aggregate (valid mean +7.81% → +9.49%, positive rate 56.7% → 67.8%) but
did NOT fix the cross-fold cliffs it was designed to attack — the
sawtooth is sharper, not flatter. The pre-registered prediction (lower
ceiling / higher floor) was half right: floor rose, but ceiling also
rose. **Standing conclusion: sustained exploration is helpful but not
curative for the regime-lock-in problem.** Next: MC sweep (3 seeds
te=-5 vs te=+1) to confirm the delta is above seed noise before
trusting the ranking or testing te=+3. te=+3 deferred until then.

**UPDATE — MC sweep DONE (see Runs section). te=+1 CONFIRMED real**
(beats baseline >1σ on Sharpe + positive rate; gains concentrated in
folds 7/8/9, NOT the fold-6 lottery which turned out to be seed noise,
CV~0.5). te=+3 (run 3b) is now unblocked and justified. Standing view
unchanged: helpful but not curative — cliffs remain, regime-conditioning
is the structural lever.

The remaining open question this raises: if exploration improves
within-regime ceilings but can't bridge regime boundaries, the cliffs
may be irreducible for a single fixed policy — which points back at the
regime-conditioning / regime-embedding idea as the structural fix
rather than any SAC hyperparameter.

### Other items from v7_handoff

The v7_handoff lays out an indicative run sequence (run 1 baseline →
run 2 per-ticker features in observation → run 3 portfolio-state window
→ run 4 cumulative trajectory dims → run 5 turnover penalty → possible
6+). Not duplicated here. As runs land, each gets its own writeup
in this dev log under standard `## Run N — [name] ([VERDICT])` headers.

---