# v7 Handoff

Starting point for v7. v6 is closed as of this writing
(see `dev_log_v6.md` for the full history). This doc captures the v7
plan, what carries forward from v5/v6, what v6 actually taught us, and
the limitations to be aware of going in.

---

## v7 in one sentence

Stop trying to predict individual stock returns. Pivot to **allocation across
a small fixed basket of 5 hand-picked tickers.**

## Why this is the right pivot

Single-ticker stock-picking has a known signal-to-noise ceiling — individual
stock returns are dominated by idiosyncratic noise that no amount of feature
engineering can extract. Sector/asset-class allocation has a meaningfully
higher ceiling because most of what successful systematic equity strategies
actually do is decide *what fraction of capital to put where*, not *which
single stock to time*.

**v6 confirmed this empirically.** v6 capstoned at +10.47% return / +0.484
Sharpe (3-seed MC mean), versus v5 baseline of +9.18% / +0.485. That's
+1.29pts return at preserved Sharpe across six v6 ablations (one architectural
simplification + three feature additions + two rejected ablations). The
ceiling for single-ticker SAC timing at v6's data scale is right around
this number. v6 hit it, and adding more architecture/feature work within
v6's framing wasn't going to materially exceed it.

The pivot to v7 is **not** because v6 is hopeless — it produced a real,
defensible result. The pivot is because v7 attacks a fundamentally easier
problem (allocation across few assets) with structurally higher upside.
Most of the remaining alpha lives there.

## Why 5 tickers specifically

The sweet spot for several constraints simultaneously:

- Small enough that action space stays tractable (5-dim continuous, well
  within SAC's comfort zone).
- Big enough for meaningful diversification benefits.
- Realistic for personal-account deployment (5-15 positions is typical).
- Cheap enough to iterate (5 tickers × 22 years is small enough to train
  fast and run lots of MC sweeps).

## Likely basket compositions

Pick one when v7 actually starts:

- **Asset-class diversity**: SPY, TLT, GLD, USO, UUP — stock/bond/gold/oil/
  dollar. Cleanest macro story; v5/v6 regime features (breadth, VIX, RS-vs-
  SPY, yield curve, credit spread, etc.) map directly. **Default lean.**
- **Sector slice**: XLK, XLF, XLE, XLV, XLY — pure sector rotation across
  major economic sectors.
- **Risk-on/risk-off**: SPY, QQQ, GLD, TLT, IWM — captures the main
  allocations a real systematic equity strategy would shift between.

Default lean is asset-class diversity because the regime feature library is
designed with macro signals in mind.

---

## What carries forward from v5/v6

Most of the infrastructure ports over directly:

- **Preprocessing pipeline** — feature engineer (now refactored from one
  3079-line file into 8 helper modules + 2298-line orchestrator), auditor,
  regime feature computations, lifecycle-segment + discontinuity-gate
  handling. The `regime_features.py` module exists and is wired in but
  most of its content (the 3a additions) didn't make v6 capstone — it
  stays in case future regime feature work wants to start from there.
- **Walk-forward validation framework** — 9-fold sequential, train/valid
  split per fold, the per-fold MC discipline we landed on in v6.
- **Trainer skeleton** — episode loop, validation cadence, logging,
  plotting, checkpoint discipline.
- **Determinism setup** — cudnn deterministic, full RNG coverage. Stays on.
- **3-seed MC standard** — seeds 42/43/44 per condition for every ablation.
  This is the methodology investment v6 paid for; v7 inherits it free.
- **`TickerData.columns` attribute + `SHARED_REGIME_PREFIXES` constant**
  — the debugging/auditing infrastructure that caught a silent classifier
  bug in v6. Keep both.
- **The dev-log discipline** — single-variable changes per run, pre-commit
  decision rules in writing before training kicks off, document rejections
  as cleanly as acceptances, no bundling unrelated changes.

## v6 final config (the v7 starting state)

The v6 capstone architecture is:

- **Market path**: 3-layer MLP (`F → 128 → 128 → 128`, GELU + LayerNorm)
  on the last timestep of the (60, F) market window. Replaces the
  earlier CNN+Transformer encoder. Output dim 128 (same as old encoder's
  `2 * d_model`), so Actor/Critic fusion is unchanged.
- **Per-ticker features**: ~37 features per ticker including the 4a deltas
  (5 medium-horizon delta_20 features) and 4c long-history additions
  (5 long-window base features + 2 long-horizon delta_60 features).
- **Regime path**: unchanged from v5 — breadth, VIX term structure,
  RS-vs-SPY at multiple horizons, all with deltas. The 3a macro regime
  additions (yield curve, credit spread, size factor, growth factor,
  sector rotation) are in the codebase but NOT in the v6 capstone config —
  run 3a was rejected.
- **Portfolio state**: 7 dims (Sharpe, win_rate, position info, etc.) —
  same as v5 run-5. Both subtractive ablations were rejected in v6.
- **Replay buffer**: `IndexReplayBuffer` with `decay=3.0` recent-emphasis
  sampling. **Flagged for v7 review** (see open design questions).
- **Training**: 200 episodes per fold, SAC with UTD=2, slow alpha LR=1e-5.

v6's 4b architecture (MLP, not encoder) is the relevant prior for v7 —
allocation problems don't obviously need sequence modeling of the per-
ticker feature space when the features themselves already encode time.
v7 should start with the MLP architecture and only revisit if there's
a specific reason.

## What v6 taught us (diagnostic conclusions worth carrying forward)

Five findings from v6 that should inform v7 design:

1. **Subtractive portfolio_state ablations are closed.** v6 runs 1 (Sharpe)
   and 2 (win_rate) both rejected unanimously. "Math-level pathology"
   (clipping saturation, bimodal distribution) does not predict "information
   content uselessness." Future portfolio_state work in v7+ should be
   **additive only** — adding new dims, not removing existing ones.

2. **Broad-spectrum macro regime features don't transfer to single-
   ticker timing.** v6 run 3a rejected 5-family macro feature bundle
   (yield curve, credit spread, etc.). The agent's decision is "trade
   THIS ticker right now," not "what's the market regime" — macro context
   has small marginal value for that specific question. For v7 (where
   the agent IS making regime-level allocation decisions), the same
   features may actually pull weight. Worth testing later in v7.

3. **The CNN+Transformer market encoder wasn't doing enough temporal work
   to justify its cost.** v6 run 4b. MLP-on-snapshot matches encoder at
   ~63% params, ~67% training time. The encoder contributed ~0.01
   Sharpe/fold of uniform temporal work — real but small. For v7,
   default to MLP unless there's a specific reason to revisit.

4. **Long-history per-ticker context matters.** v6 run 4c added 120/252-day
   base features with 60d deltas (5 base + 2 deltas) and gained +0.76pts
   return / +0.028 Sharpe over the 4b baseline. The "long base + medium
   delta" pattern from v5 run-3 generalizes. For v7, this means the
   per-ticker feature set should include multi-quarter horizons, not just
   ≤60-day windows.

5. **Fold 7 (2021 melt-up) is a recency-emphasis problem, not a feature
   or architecture problem.** **FIVE** qualitatively different v6 runs
   (1, 3a, 4a, 4b, 4c) all hurt fold 7 disproportionately. The cause is
   upstream: `IndexReplayBuffer` with `decay=3.0` weights the most
   recent training year disproportionately, producing chaos-trained
   reflexes that underperform 2021's slower melt-up dynamics. This is
   the single most important v6 finding for v7 to address. See open
   design question 6.

## What changes for v7

- **Environment**: 5-dim continuous action space (target dollar allocation
  per ticker, summing to ≤ 100% of capital with cash as implicit residual).
  Portfolio-level state tracking. Weight constraints. Probably keep the
  dollar-deadband logic but apply per-ticker.
- **Networks**: actor outputs 5-dim vector. Critic takes 5-dim action.
  Start with v6's MLP-on-snapshot architecture (4b answered the encoder-
  vs-MLP question for v6's single-ticker setup; same answer probably
  ports). **Open design question**: share market encoder across 5 tickers,
  or run 5 parallel encoders? Shared is cheaper and forces representation
  generalization across tickers; parallel is more expressive but 5× the
  parameter count.
- **Reward**: joint portfolio return rather than single-ticker return.
  Reward shaping decisions become more interesting in v7 — sharpe-adjusted
  vs raw return, drawdown penalties at portfolio level, turnover penalties
  to discourage churn.
- **Data**: 5 specific ticker columns from unified.parquet instead of the
  one-ticker-per-episode sampling. Episodes become "5-ticker portfolio
  over a 252-day window" rather than "single ticker over 252 days." The
  episode count drops by ~3000× since we're not sampling across tickers
  anymore — need to think about what episode structure replaces that
  diversity. Probably overlapping start-date windows.
- **Action smoothing / turnover**: real allocators don't reallocate every
  day. Worth thinking about whether to add a rebalancing schedule
  constraint (weekly? monthly? based on action magnitude threshold?) or
  let the policy learn rebalancing frequency on its own with a turnover
  cost in the reward.

## Open design questions for v7

These need decisions before run 1:

1. **Observation structure**: per-ticker features for all 5 in observation,
   or only aggregated regime features + per-ticker positions? First is more
   expressive but harder; second is cleaner but throws away information.
   Lean toward first with shared encoder.
2. **Encoder topology**: shared vs parallel (above).
3. **Portfolio-state window**: see "Portfolio-state window" section below.
   This is the v6-era idea that becomes high-EV in v7 because portfolio
   state expands to ~15-25 dims.
4. **Rebalancing constraint**: hard schedule, soft turnover penalty, or
   neither (let the policy figure it out).
5. **Initial allocation**: start each episode at all-cash, equal-weight,
   or sampled from prior policy state? Equal-weight is the boring-correct
   choice; all-cash forces the policy to demonstrate active value
   add over a do-nothing baseline.
6. **Replay buffer recency emphasis (`decay`)**: v6 uses `decay=3.0`
   recent-emphasis sampling, meaning ~60% of gradient updates come from
   the newest third of stored transitions. v6 post-mortem on the fold-7
   regression strongly suggests this was too aggressive — the policy at
   each fold's training end was dominated by whatever the most recent
   training years happened to be, which created a fold 6 (best fold)
   vs fold 7 (worst fold) mismatch driven by recent-regime composition
   rather than feature/architecture choices. See dev_log_v6.md's
   "Correction — fold-year mapping" section for the full diagnosis.

   For v7's fixed 5-ticker basket the buffer dynamics are very different
   (no cross-ticker mixing within episodes, much smaller absolute buffer
   size, more transitions per ticker per training step). Worth running
   a sweep across decay values early in v7 — likely candidates {0.0
   uniform, 1.0 mild emphasis, 3.0 v6 default}, single fold to start
   for cheap feedback, then commit to the chosen value before running
   the actual MC ablations. Could be worth its own pre-run-1
   investigation since it affects every subsequent run's interpretation.

---

## Portfolio-state window — high-EV in v7

Built up during v6 thinking, deferred from v6 because the marginal value
was small for single-ticker. v7 is where this actually earns its cost.

Each feature group enters the network with a different path-dependence
mechanism in the v6 capstone:

| Group               | Mechanism                                  |
|---------------------|--------------------------------------------|
| Market features     | MLP on last-timestep snapshot + explicit deltas (4a/4c)  |
| Regime features     | MLP on snapshot + explicit deltas           |
| Portfolio state     | MLP on snapshot, no deltas, no window       |

Portfolio_state is the odd one out — no temporal mechanism. In v6 this was
fine because there were only 6-7 portfolio dims and most weren't trajectory-
rich. In v7 portfolio state will be substantially richer (5 weights, 5 per-
asset unrealized P&Ls, total exposure, possibly per-asset hold times — easily
15-25 dims), and allocation problems are inherently path-dependent
(rebalancing, drift, momentum-in-weights, recent volatility per position).

**Recommended approach**: window over the last N days of all portfolio_state
dims, same way market features get a window. Window length probably much
smaller than market's 60d — portfolio dynamics move slower. Candidate values:
5, 10, 20.

**Architectural options:**

1. **Flatten + MLP**: treat (W_p × P) matrix as a flat vector, run through
   wider MLP. Cheapest. Loses sequential structure but minimal codebase
   change. Right first try.
2. **Small conv or attention**: model the time axis explicitly. Worth
   trying only if option 1 shows signal.
3. **RNN**: overkill for W_p ≤ 20. Skip.

**Replay buffer impact**: `DailyReplayBuffer._state_dim` grows by
`(W_p - 1) * P` if storing flattened. Update `_flatten` and `_unflatten_batch`
to handle the window shape. Existing v6 buffers become incompatible — fine,
v7 is a clean break anyway.

**Env impact**: `_get_observation` maintains a rolling buffer of the last
W_p portfolio snapshots. Pre-episode, fill with zeros (analogous to market-
data zero-padding for the first window_size steps).

### Related portfolio_state additions worth bundling

If a portfolio_state overhaul lands in v7, these touch the same code and
should be considered together:

- **Cumulative trajectory dims**: realized P&L so far, fraction of episode
  spent invested, max position size reached, drawdown peak. New info about
  agent behavior across the episode beyond what's in any single snapshot.
- **win_rate stays.** Tested in v6 run 2 (3-seed MC), all seeds worse,
  reverted. The metric's bimodality is real but the network was using it
  productively. Don't remove it during v7 portfolio_state work — the
  temptation will come up but the v6 ablation answered the question.

A v7 portfolio_state design pass is an architectural change, not a single-
variable ablation. Frame and document accordingly.

---

## Known limitations carried forward from v6

### Universe-selection survivorship bias

**This is the most important known limitation. Read it carefully before
comparing v7 results against v6.**

The discontinuity gate, liquidity filter, MIN_TICKER_LENGTH gate, penny-
stock floor, and ticker-events filtering all run **once, on the full data
history (2003-2026), before any walk-forward split happens** in v6. The set
of tickers that survive into the universe is determined using information
from the entire timeline, including periods that are walk-forward validation
periods.

This means v6's universe is biased toward tickers that survive 22 years of
filtering. Effects:

- A ticker that has a critical discontinuity in 2024 is excluded from
  training in 2010, even though the model would legitimately have seen and
  traded it at that time.
- A ticker that becomes illiquid in 2022 is dropped for low average dollar
  volume (averaged across the full history). The model never trains on it,
  even though it was liquid in 2015.
- Tickers that delist mid-training-period get dropped retroactively for
  MIN_TICKER_LENGTH violations.

Net effect: every v6 fold trains and validates on a universe pre-selected
to survive the entire timeline. This is the same shape as survivorship bias
in academic studies that use today's S&P 500 constituents to study historical
returns.

**Why v7 mostly fixes this for free**: v7's hand-picked 5-ticker basket
sidesteps the universe-selection question entirely. The universe is fixed
by design — SPY, TLT, GLD, USO, UUP are all going to be there for the
full timeline. No discontinuity gate, no liquidity filter to apply. Universe
survivorship goes away as a category of bias.

**The risk worth being explicit about**: this means v7 results are *not*
directly comparable to v6 results in a way that lets you say "v7 beat v6
by X." v6 had an unfair advantage from universe survivorship. To do a fair
v6-vs-v7 comparison, you'd need to either:

1. Rerun v6 with per-fold universe filtering (expensive — 9× preprocessing).
2. Accept that v7 is solving a different problem and compare against a
   buy-and-hold-of-the-basket baseline instead.

Option 2 is the right framing. v7's reference point isn't v6; it's the
benchmark a real allocator would compare against — equal-weight buy-and-
hold of the basket, or SPY-only as a single-asset reference. This is also
the framing real systematic strategies use.

**To preempt the confusion**: if v7 produces Sharpe 0.6 mean across folds,
that is *not* "v7 is barely better than v6 (0.48)." The two numbers are
measuring different things on different universes. v7 needs to clear
its own bar (basket buy-and-hold) by a meaningful margin to be
worth deploying.

### Other limitations inherited from v6

- **Polygon vendor data gaps**: ~14% of v6's universe is lost to the
  discontinuity gate because Polygon's splits endpoint is incomplete (GOOG
  2014, EFA/EEM unrecorded splits, etc.). v7's hand-picked basket sidesteps
  this for the chosen 5 tickers but the underlying data quality issue
  remains for any future universe expansion.
- **Databento migration**: pre-staked as a v7+ data quality improvement.
  Cleaner historical corporate actions would recover the lost 14% of the
  v6 universe and make the universe-survivorship fix above cheaper to do
  per-fold. Most useful paired with v7+ work since v7's small basket
  doesn't need it; bigger v8+ universes would.

---

## v7 run sequence (proposed)

Use the v6 single-variable-per-run + 3-seed MC discipline throughout.

**Pre-run setup (real work before the first MC sweep):**

- Pick the 5-ticker basket.
- Decide initial allocation (recommend equal-weight or all-cash).
- Decide rebalancing constraint (recommend "none, let policy decide, with
  turnover cost in reward").
- Decide encoder topology (recommend shared encoder, single test in run 2
  if signal is ambiguous).
- **Recency-emphasis sweep**: cheap pre-MC investigation across `decay`
  values to commit to the right setting for v7 before running ablations.
  v6's five-run fold-7 confirmation makes this the highest-value pre-run
  decision. Likely candidates {0.0 uniform, 1.0 mild, 3.0 v6 default};
  single fold for cheap feedback, lock in the choice, then run ablations
  against that locked baseline.

**Run 1: baseline 5-ticker setup.**
Match v6 capstone architectural defaults as closely as the multi-ticker
action space allows (MLP-on-snapshot, no encoder; 4a deltas + 4c long-
history features per ticker; chosen recency-emphasis from pre-run setup).
Establish the v7 baseline against the buy-and-hold reference.
Single-variable change for the run is "single-ticker → 5-ticker portfolio."

**Run 2: per-ticker features in observation.**
If run 1 baseline doesn't have per-ticker features (e.g. only regime
features), add them with shared encoder. The first encoder-topology test.

**Run 3: portfolio-state window.**
The high-EV trajectory-info addition described above. Window size W_p
probably 10 as a starting point.

**Run 4: cumulative trajectory dims in portfolio state.**
Realized P&L, fraction time invested, max position size. Adds to the
window dims from run 3.

**Run 5: turnover penalty in reward.**
If runs 1-4 show high churn, add a turnover cost.

**Possible run 6+: revisit macro regime features.**
v6 run 3a rejected the macro regime bundle (yield curve, credit spread,
size factor, growth factor, sector rotation) for single-ticker timing.
But v7 is an allocation problem — macro context may genuinely help the
"which asset class to weight" decision that single-ticker SAC didn't
benefit from. Worth testing late in v7 once the baseline is solid.

After that, depends on what 1-5 reveal. The v6 lesson — that mid-run
intuitions about what to try next often beat the original plan — applies.

---

## Useful context to bring forward

- `dev_log_v5.md` and `dev_log_v6.md` — full history of what was tried and
  rejected. v6 dev log is particularly worth reading the "Correction —
  fold-year mapping" section near the top (covers the fold-7 recency-
  emphasis diagnosis) and the closing "v6 Capstone" section.
- `preprocessing.md` — full pipeline documentation with vendor caveats
  and known data limitations.
- `v6_handoff.md` — the original v6 plan, kept frozen as historical
  record (don't edit; it preserves the mentality going into v6).
- Reference distributions to bring forward:
  - v5 (single-ticker, 3-seed MC): ret_mean +9.18% ± 3.41, sh_mean +0.485
    ± 0.052.
  - v6 capstone (single-ticker, 3-seed MC): ret_mean +10.47% ± 3.91,
    sh_mean +0.484 ± 0.092.
  - **Neither is directly comparable to v7** due to universe survivorship
    above. Kept as historical context only. v7's reference point is
    basket buy-and-hold, not v5/v6 numbers.

## Long-term vision

```
v5 (done):    single-ticker, clean data, regime features, UTD ablation
              ended at: validation Sharpe 0.485 mean, ~62% positive, peak fold Sharpe 1.0+

v6 (done):    single-ticker architectural + feature experiments
              capstone:  v5 + per-ticker deltas + MLP-only encoder + long-history features
              ended at: validation Sharpe 0.484 mean, ~60% positive
                        (+1.29pts ret over v5 at preserved Sharpe)
              key finding: fold-7 problem is recency-emphasis in replay buffer,
                           not features or architecture; 5-run confirmed

v7 (next):    5-ticker portfolio, allocation problem,
              clean universe (no survivorship), addresses recency emphasis

v8+:          full sector rotation or larger universe allocation
              (this is where Databento migration likely earns its keep)
```
