# v6 Handoff Summary

For starting the v6 chat from a clean context.

---

## Where v5 ended

Five runs total, single-variable change per run.

- **Run 1**: catastrophic outliers, found 2 hidden data bugs (lifecycle gaps + close-null filling). Fixed.
- **Run 2**: clean baseline. Validation: +1.87% mean, 0.32 Sharpe, 59.4% positive.
- **Run 3**: long-horizon regime features (`breadth_200d`, `rs_spy_252d` + medium deltas). Validation: +4.11% mean, 0.36 Sharpe, 77.8% positive.
- **Run 4**: UTD=4 + slower alpha LR. Overfit. Validation: +3.04% mean, 0.30 Sharpe, 54.4% positive. Useful negative result — proved gradient-update density past a threshold hurts generalization.
- **Run 5**: UTD=2. **The v5 capstone.** Validation: +8.52% mean, 0.47 Sharpe, 74.4% positive. Peak fold-6 Sharpe sustained above 1.0. Fold 5 (historically the hardest) went from -10% (run 3) to +24% (run 5).

UTD ablation across {1, 2, 4} produced a clean three-point dose-response: UTD=1 underfits, UTD=2 is the sweet spot, UTD=4 overfits. UTD=2 is the v6 baseline going forward.

v5 core goal achieved: model trained on clean data learns a real signal that generalizes across regimes, with a meaningful improvement (+1.87% → +8.52% validation mean) from systematic per-run iteration.

---

## v6 theme

Architectural and structural experiments to push past the v5 ceiling. **v6 baseline is run 5's config: UTD=2, slow alpha LR (1e-5), all run-3 features.** Starting point is validation Sharpe 0.47 mean, 74% positive rate, with peaks above 1.0 in good folds. v6 needs to beat that to be worth the effort.

## v6 TODOs (proposed run order)

### Small wins first

**1. Remove Sharpe from portfolio_state**
- Real footgun. Per-step Sharpe clipped to [-3, 3] makes "real high performance" and "degenerate near-zero activity" indistinguishable to the network. Could reinforce undertrading.
- Change is in `environment.py` `_get_observation`. Reduces portfolio_state by 1 dim. Sharpe stays in `_get_info` for logging.
- Breaks checkpoint compatibility with v5 — clean break, fine for v6.

**2. More regime features from unused regime tickers**
- ~25 regime tickers loaded but only SPY/VIXY/VIXM contributing meaningfully.
- Candidates: sector rotation (XLF, XLK, etc.), yield curve (TLT/SHY), credit spread (HYG/LQD), dollar regime (UUP), commodities (GLD, USO), international RS (EFA, EEM, EWJ).
- Higher expected value than another long-horizon-of-same-thing addition.

### Bigger experiments

**3. Per-ticker architecture: encoder vs MLP+deltas**
- Current: 60-day window through CNN+Transformer encoder.
- Alternative: snapshot+deltas through MLP. Past-me wrote this TODO in `networks.py`.
- PREREQUISITE: add per-ticker delta features in `feature_engineer.py` first.

**4. Replay buffer warmup before gradient updates**
- First 50-100 episodes fill replay only, no gradient updates, fixed alpha=0.2.
- Skips the high-alpha chaos at start of training where alpha collapses 0.99 → 0.025 in ~150 episodes.

**5. Portfolio blowup early-termination**
- TODO already commented in `environment.py`. Just uncomment + add `MAX_DRAWDOWN_FRAC` constant.
- `if current_portfolio_value <= initial_balance * (1 - 0.30): done = True`
- Drawdown from initial, not peak.
- Watch `avg_episode_length` once enabled; early terms skew replay buffer toward early-episode transitions.

### Infrastructure (do alongside the work above)

**6. Determinism hardening for clean ablations**
- `random.seed(SEED)`, `torch.cuda.manual_seed_all(SEED)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- Lock column order in unified parquet so re-preprocessing doesn't shuffle features.

## Pure housekeeping (no version, do anytime)

- **File-split refactor of `feature_engineer.py`** (~3000 lines). Plan documented in v5 dev log. Bottom-up move starting with `constants.py`, `sectors.py`, `calendar.py`. No logic changes.
- **Centralize `DISCONTINUITY_LOG_THRESHOLD`** in config.py (currently in feature_engineer.py with reference comment in auditor.py).

---

## Deferred to v7

**Multi-asset trading: 5 hand-picked tickers.**

The conceptual shift v7 represents: stop trying to predict individual stock returns (high noise, low signal), pivot to allocation across a small fixed basket. Single-ticker stock-picking has a known ceiling because individual stock returns are mostly unpredictable. Sector/asset-class allocation has a meaningfully higher ceiling — this is what most successful systematic equity strategies actually do.

**Note on the ceiling**: run 5's Sharpe 0.47 mean (with peaks above 1.0 in good folds) is higher than the originally-assumed single-ticker ceiling. v6 architectural work might push this further. The pivot to v7 isn't because v6 is hopeless — it's because v7 attacks a fundamentally easier problem with higher upside. Both are worth doing.

5 tickers is the sweet spot:
- Small enough that action space stays tractable (5-dim continuous, well within SAC's comfort zone)
- Big enough for meaningful diversification benefits
- Realistic for personal-account deployment (5-15 positions is typical)
- Cheap enough to iterate (5 tickers × 22 years is small enough to train fast)

**Likely compositions** (pick one when v7 starts):
- **Asset-class diversity**: SPY, TLT, GLD, USO, UUP — stock/bond/gold/oil/dollar. Cleanest macro story, regime features (breadth, VIX, RS-vs-SPY) map directly.
- **Sector slice**: XLK, XLF, XLE, XLV, XLY — pure sector rotation across major economic sectors.
- **Risk-on/risk-off**: SPY, QQQ, GLD, TLT, IWM — captures the main allocations a real strategy would shift between.

Lean toward asset-class diversity because v5/v6 regime features were designed with macro signals in mind.

**What v7 preserves from v5/v6:**
- All preprocessing infrastructure
- Feature engineering pipeline (regime features especially carry over directly)
- Audit pipeline
- Walk-forward validation framework
- Trainer skeleton
- Logging/plotting

**What changes for v7:**
- **Environment**: 5-dim action space, portfolio-level state tracking, weight constraints (probably target dollar values that sum to ≤ 100% of capital, with cash as implicit residual)
- **Networks**: actor outputs 5-dim vector. Critic takes 5-dim action. Open question whether to share encoder across 5 tickers or use 5 parallel encoders.
- **Reward**: still portfolio return, but now joint return of 5-asset portfolio
- **Data**: 5 specific ticker columns instead of one-ticker-per-episode sampling

**Open design question for v7**: should agent see per-ticker features for all 5 tickers in observation, or only aggregated regime-level features? First is more expressive but harder; second is cleaner but throws away information. Lean toward first with shared encoder.

### Portfolio-state window + architectural support for it

Origin: 2am thought during v6 run 1. Started as "should portfolio_state get delta features like regime does?" and resolved into something more interesting.

Each feature group currently enters the network with a different path-dependence mechanism:

- Market features (~25 dims) → 60-day window through CNN+Transformer encoder.
- Regime features (shared) → MLP on snapshot + explicit deltas.
- Portfolio_state → MLP on snapshot, no deltas, no window. **Odd one out.**

In v6 single-ticker, portfolio_state has only 6-7 dims and the marginal value of trajectory info is small — there isn't much state to model the history of. In v7 the portfolio_state will be substantially richer (5 weights, 5 per-asset unrealized P&Ls, total exposure, possibly per-asset hold times — easily 15-25 dims), and allocation problems are inherently path-dependent (rebalancing, drift, momentum-in-weights). Trajectory info on portfolio state has high value in v7 specifically.

**Why deltas alone aren't the right fix**: portfolio_state dims decompose into action-history dims (cash/stock weights, hold time — deltas redundant with the agent's own action stream), price-action-while-held dims (unrealized P&L, drawdown — deltas would carry real info), and statistical accumulator dims (win_rate, etc. — bimodal at episode level, deltas inherit the pathology). Picking and choosing per-dim is fiddly.

**Cleaner approach**: window over the last N days of all portfolio_state dims, like market features get a window. Window length probably much smaller than market's 60d — portfolio dynamics move slower. Candidate values: 5, 10, 20.

**Architectural options for the encoder:**

1. Flatten + MLP. Treat (W_p × P) matrix as a flat vector, run through wider MLP. Cheapest. Loses sequential structure but minimal codebase change. Right first try.
2. Small conv or attention. Models the time axis explicitly. Worth trying only if option 1 shows signal.
3. RNN. Overkill for W_p ≤ 20, skip.

**Replay buffer and env impact:**

- `DailyReplayBuffer._state_dim` grows by `(W_p - 1) * P` if storing flattened. Update `_flatten` and `_unflatten_batch` to handle the window shape. Existing buffers in checkpoints become incompatible (clean break, fine for v7).
- `_get_observation` maintains a rolling buffer of the last W_p portfolio snapshots. Pre-episode, fill with zeros (analogous to market-data zero-padding for the first window_size steps).

### Related v7+ portfolio_state work

Consider batching with the window change since they touch the same code:

- **Cumulative trajectory dims**: realized P&L so far, fraction of episode spent invested, max position size reached. New info about agent behavior across the episode beyond what's in current snapshot.
- **Removing win_rate**: pre-staked TODO from v5. Bimodal at episode level, low SNR. Would have been a v6 candidate, but the Sharpe-removal lesson (run 1) suggests treating "obvious footguns" with skepticism — the metric might still be doing useful work the math doesn't capture. If touched, do it under MC.

If a portfolio_state overhaul run happens, batching these makes sense — they share code paths and are all "portfolio_state content" decisions. But that's an architectural redesign, not a single-variable ablation, and would need to be framed as such.

## Deferred to v7+ or later

- **Databento migration.** ~14% of universe currently lost to discontinuity filter due to Polygon's incomplete splits coverage. Cleaner data would recover those. Most useful when paired with v7 redesign anyway since you'd want clean data for the multi-asset version.
- **Ensemble / regime-classifier architecture.** Final-version material.

---

## Useful context to bring forward

**Files (in repo):**
- `feature_engineer.py` (~3000 lines) — all data filtering upstream, trainer is pure consumer
- `auditor.py` — verifies, doesn't filter
- `trainer.py` — vertical log format, per-run dirs, incremental saves
- `environment.py` — single-ticker-per-episode, dollar deadband, fractional shares, EOD liquidation. Has TODO comments for portfolio blowup early-termination.
- `agent.py` — has `_gradient_step` helper from v5 run 4 refactor
- `networks.py` — TODOs already in place for architecture experiments
- `preprocessing.md` — pipeline doc with vendor caveats and known data limitations
- `dev_log_v5.md` — full v5 history

**Working principles internalized during v5:**
- Single-variable changes per run. Bundling kills attribution.
- Filter upstream, trainer consumes clean data, auditor verifies.
- Run 1 of any new fold can look bad and not be a real problem. Don't kill on fold 1 alone.
- Within-fold trajectory matters as much as fold-end snapshot.
- Verify config against actual GitHub commits before assuming working-copy state matches.

**Polygon data caveats already known:**
- ~1100 suspect 50-100% one-day moves (unapplied splits Polygon doesn't have, e.g. GOOG 2014-04-03 Class C creation)
- Decision: live with it for v5/v6, fix via Databento in v7.

---

## Suggested v6 run sequence

Baseline config carried forward from v5 run 5:
- UTD=2
- LEARNING_RATE_ALPHA = 1e-5
- All run-3 regime features
- All run-2 data fixes

v6 runs each add one variable on top:

- Run 1: remove Sharpe from portfolio_state
- Run 2: + new regime features (sector rotation, yield curve, etc.)
- Run 3: + per-ticker delta features (preprocessing change, no model change yet)
- Run 4: + MLP-only architecture for per-ticker path
- Run 5: + replay buffer warmup
- Run 6: + portfolio blowup early-termination

Each adds one variable on top of the previous. If any one fails to help, revert and continue building from the last known-good config.

---

## Long-term vision

```
v5 (done):    single-ticker, clean data, regime features, UTD ablation
              ended at: validation Sharpe 0.47 mean, 74% positive, peak Sharpe 1.0+
v6 (now):     architectural experiments built on UTD=2 baseline
v7:           5-ticker portfolio, real allocation problem
v8+:          full sector rotation OR full single-name universe with allocation
```

The honest framing: v5 finished stronger than expected — UTD=2 was an unexpected win. v6 builds on that with structural improvements that should compound. v7 pivots to a fundamentally different problem (allocation, not timing) that's known to have higher signal-to-noise. v8+ depends on what v6 and v7 reveal.