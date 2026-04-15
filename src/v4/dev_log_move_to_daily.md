# Dev Log: Minute-Level Training Conclusion & Move to Daily

## Date: April 10, 2026

---

## Summary

After multiple training runs with progressive fixes to the SAC training pipeline, minute-level TSLA trading has not produced a viable policy. The architecture, preprocessing, and training mechanics are validated — the issue is signal-to-noise ratio at the minute resolution. Moving to daily bars for the next iteration.

---

## Training Runs Attempted

### Run 1: Delta Action Space, target_entropy=-0.5
- Alpha collapsed to 0.0 by episode 25
- Entropy reached target (-0.5) and flatlined
- Returns converged to ~0% — agent learned to sit in cash
- **Root cause:** Delta action space forced alpha collapse (agent needed deterministic output to hold positions)

### Run 2: Delta Action Space, target_entropy=-0.1
- Identical alpha collapse trajectory — alpha at 0.0 by episode 25
- Changing target entropy alone did not fix the structural action space problem
- Confirmed the issue was the action space, not the entropy target

### Run 3: Target Position Action Space, target_entropy=-0.1, alpha_floor=0.05
- Alpha floor prevented collapse — alpha held at 0.05
- But entropy plateaued at ~0.667 for 140 episodes — never approached target of -0.1
- All train metrics flatlined: returns at -20%, trade count at ~1,700, win rate at 25%
- **Diagnosis:** Floor too high — created equilibrium where exploration noise perfectly balanced Q-value gradient, preventing policy refinement

### Run 4: Target Position Action Space, target_entropy=-0.1, alpha_floor=0.01
- Best run. Entropy dropped from 0.68 to 0.33 over 100 episodes
- Train metrics showed real improvement through episodes 50-80:
  - Returns MA: -22% → -11%
  - Win rate MA: 25% → 42%
  - Max drawdown MA: 22% → 12%
  - Fee impact MA: 0.55% → 0.30%
  - Trade count: 1,780 → 1,480
- Validation: 47% positive returns (7/15), beat buy-and-hold 40% of the time
- **But:** All improvements reversed after episode 80-100
  - Returns MA climbed back from -11% to -13% by episode 150
  - Critic loss rose from 0.0025 to 0.01 (policy instability)
  - Entropy plateaued at 0.36 — exploration noise still producing σ ≈ 0.22 on [-1,1] scale
  - Train trade count still ~1,500/episode at episode 150 — mostly noise trades
  - 0 out of 150 train episodes had positive returns (0.0%)

---

## Evidence: Signal-to-Noise Problem

### The math on exploration noise
- Entropy at 0.36 corresponds to σ ≈ 0.22 on [-1, 1] action space
- With max_trading_units=30, the deadband absorbs ±2 shares (±0.13 in action space)
- σ of 0.22 means 32% of exploration noise exceeds the deadband → triggers unwanted trades
- Even at target entropy of -0.1 (σ ≈ 0.22), the noise is massive relative to TSLA's minute-level signal

### Train vs validation gap
- Train: 0/150 episodes positive (0.0%), mean return -13%
- Validation (deterministic): 7/15 episodes positive (47%), mean return -1.4%
- The deterministic policy performs reasonably — it's the training exploration that destroys performance
- This suggests the signal exists but is too weak to learn through the noise of minute-level exploration

### Transaction costs during exploration
- Train fee impact: 0.3-0.5% per episode with ~1,500 trades
- Each exploration trade costs spread + slippage + regulatory fees
- At 1,500 trades/episode, the agent is paying ~$500-700 in fees on a $10,000 account
- Any minute-level alpha signal needs to overcome this drag just to break even during training

### What worked vs what didn't
- ✅ Architecture learns: validation shows the deterministic policy can track price movements
- ✅ Regime features: validation actively trades (200-500 trades) with some positive episodes  
- ✅ Action space: target position + deadband solved the holding problem
- ❌ Training signal: minute-level returns are too noisy for the exploration process to extract consistent patterns
- ❌ Entropy control: every floor value either over-corrects (0.05) or under-corrects (0.01) — there's no sweet spot because the fundamental issue is resolution, not hyperparameters

---

## Decision: Move to Daily Bars

### Rationale
1. **Signal-to-noise:** Daily returns have ~20x higher signal-to-noise ratio than minute returns. A 1% daily move is meaningful; a 1% minute move is noise.
2. **Faster iteration:** Episodes measured in seconds, not 10 minutes. Can run 1,000 episodes in hours instead of days.
3. **Architecture validation:** If the model can't learn on daily, the issue is fundamental (features, reward, network design). If it works on daily but not minute, the issue is confirmed as resolution/noise.
4. **Simpler exploration:** With ~252 steps/year vs ~98,000 steps/year, each action has more impact and the exploration-exploitation tradeoff is far more tractable.
5. **Transaction cost clarity:** Daily trades are 1-5 per episode vs 1,500+ on minute bars. Fee impact becomes negligible, letting the model focus on direction.

### What carries over
- Target position action space with proportional deadband
- Regime feature separation (MLP) vs market features (FeatureExtractor)  
- Alpha floor mechanism
- Batch prefetching
- Reward clipping
- Preprocessing pipeline (feature engineering, cross-sectional normalization, AE compression)
- All need adaptation for daily resolution but the design patterns are validated

### What changes
- Window size: 120 minutes → TBD trading days
- Episode length: ~1,950 steps (5 trading days) → TBD
- Feature windows: 5/15/60 minute → daily/weekly/monthly
- NORMALIZATION_WINDOW: 3,900 bars → TBD
- max_trading_units: may need adjustment for daily position sizing
- Target entropy: -1.0 (validated as appropriate for 1D continuous action with target position)
