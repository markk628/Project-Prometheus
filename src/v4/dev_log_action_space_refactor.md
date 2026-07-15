# Dev Log: Action Space Refactor & Alpha Collapse Fix

## Date: April 7–9, 2026

---

## Problem: Alpha Collapse by Episode 30

### Symptoms
- Alpha (entropy coefficient) decayed from 1.0 to ~0.0 by episode 25-30 across multiple training runs.
- Entropy reached the target of -0.5 and flatlined.
- Returns hovered around 0% after the initial learning phase, with the agent sitting in cash.
- Adjusting `target_entropy` from -0.5 to -0.1 did not fix the issue — alpha still collapsed to 0 with identical trajectory.

### Root Cause: Order-Based (Delta) Action Space

The environment used a **delta action space** where the action value directly mapped to a buy/sell order:
- `action > 0` → buy `int(max_trading_units * action)` shares
- `action < 0` → sell `int(max_trading_units * abs(action))` shares
- `action == 0` → hold

**The "holding" problem:** SAC is a continuous policy algorithm. The actor outputs actions sampled from a Gaussian distribution. It is nearly impossible for a continuous Gaussian to output exactly `0.0` repeatedly. Any small deviation from zero triggers a trade.

**The deadband was tiny:** With `max_trading_units = 10`:
- `action = 0.099` → `int(10 * 0.099) = 0` shares (no trade)
- `action = 0.100` → `int(10 * 0.100) = 1` share (trade executes)

The "safe zone" where no trade occurs was only ±0.099 — a 10% band in the [-1, 1] action space. Any exploration noise outside this band triggered a market order with associated fees and slippage.

**The cascade:**
1. Early episodes: agent explores randomly → nearly every action triggers a trade → ~1,700 trades per episode → massive losses from random positions.
2. The agent quickly learns that random trading leads to large negative rewards.
3. To reduce randomness, the alpha optimizer drives alpha toward zero, reducing the entropy bonus in the actor's loss.
4. With alpha near zero, entropy collapses to the target within ~30 episodes — far too quickly for the agent to have discovered any profitable strategies.
5. The policy becomes near-deterministic early in training, limiting the agent's ability to explore different trading strategies for the remaining hundreds of episodes.
6. The agent still traded (validation showed active positions), but the rapid entropy collapse meant the policy was shaped almost entirely by the first 25-30 episodes of random exploration — not enough time to learn meaningful market patterns.

**Key evidence:** Fee impact was only 0.2–0.7% per episode. The -20% to -37% losses in early episodes came from bad positions, not fees. But the *mechanism* of the alpha collapse was the inability to hold positions without triggering micro-trades — the agent was forced to become deterministic just to maintain stable positions.

---

## Fix 1: Target Position Action Space

### Design Change
The action meaning changed from "how much to trade" to "where I want to be":

| | Old (Delta) | New (Target Position) |
|---|---|---|
| Action = -1 | Sell max shares | Hold 0 shares (all cash) |
| Action = 0 | Hold (if exactly 0.0) | Hold 50% of max position |
| Action = 1 | Buy max shares | Hold max shares (fully invested) |
| Holding | Must output exactly 0.0 | Output same value as current position |

**Conversion:** `target_shares = int(max_trading_units * (action + 1) / 2)`

The environment calculates the delta between target and current position and only executes trades when a position change is requested.

### Why This Fixes the Holding Problem
With target position, maintaining a position is natural. If the agent holds 15 shares and outputs `action = 0.0` (target = 15 shares), delta = 0, no trade occurs. If exploration noise shifts the action to `0.05` (target = 15.75 → 15 shares), delta is still 0. The agent can hold positions without needing deterministic output.

### Proportional Deadband
An additional deadband filter ignores small deltas caused by policy noise:

```python
deadband = max(int(max_trading_units * 0.07), 1)  # ~7% of max position = 2 shares
if abs(shares_delta) < deadband:
    return  # hold, no trade
```

This means the actor's output must change the target by at least 2 shares (for `max_trading_units=30`) before any trade executes. Small fluctuations from exploration noise are silently absorbed.

### `mask_action` Removed
The old `mask_action` function validated whether a trade was affordable before execution. With target position, the environment handles feasibility internally — clamping buys to affordable shares and sells to held shares. `mask_action` was removed from both the environment and trainer.

---

## Fix 2: Alpha Floor

### Problem
Even with target position and `target_entropy = -0.1`, alpha still decayed to ~0 by episode 30. The agent's policy entropy was above the target (entropy was still positive), so the alpha optimizer correctly reduced alpha — but it reduced it *too far*. Once at zero, even if entropy later dropped below target, alpha couldn't recover fast enough to prevent further entropy collapse.

### Solution
A floor was added to alpha so it never drops below 0.05:

```python
# In update_parameters (after alpha optimizer step):
self.alpha = torch.clamp(self.log_alpha.exp(), min=0.05)

# In load_model (to prevent loading a collapsed alpha):
self.alpha = torch.clamp(self.log_alpha.exp(), min=0.05)
```

### Effect
- Alpha decays naturally from 1.0 but stops at 0.05 instead of reaching 0.
- The actor loss always includes a small entropy bonus: `loss = 0.05 * log_probs - Q`
- The Q-value gradient dominates policy updates (exploitation), but the 0.05 entropy term provides a constant gentle pressure toward exploration.
- Entropy decreases more slowly and should settle near the target of -0.1 rather than freefalling to it.
- At inference time (`validate=True`), the policy uses `torch.tanh(mean)` with no sampling — fully deterministic regardless of alpha. The floor only affects training.

### How Alpha, Entropy, and Target Entropy Relate
- **Alpha**: weight on the entropy bonus in the actor's loss function. Controls how much the agent is *rewarded* for being random.
- **Entropy**: measures how random the policy *actually is* — the width of the action distribution. High = wide spread of actions, low = nearly deterministic.
- **Target entropy**: the desired entropy level. The alpha optimizer adjusts alpha to push actual entropy toward this target.
  - Entropy above target → alpha decreases (already random enough)
  - Entropy below target → alpha increases (need more exploration)
- **Target entropy of -0.1** does NOT mean deterministic. Fully deterministic = entropy at negative infinity. At -0.1, the policy is confident but maintains meaningful variance.
- **Alpha at 0.05 floor** means entropy has a gentle brake — it can still decrease toward target but can't freefall into full determinism.

---

## Fix 3: Regime Feature Separation & Batch Prefetching

### Regime Feature Architecture Change
Regime features (cross-sectional z-scores, RS vs SPY, sector z-scores, VIX term structure, market breadth — ~1,256 columns) were originally compressed alongside market features through the autoencoder in the dimensional reducer. They were separated out for two reasons:

1. **Different signal type:** Market features (price action, volume, volatility ratios) have meaningful temporal microstructure across the 120-bar window — patterns the CNN and Transformer in FeatureExtractor are designed to capture. Regime features are cross-sectional snapshots that change slowly and don't have tick-level sequential patterns worth modeling.

2. **Preserve regime signal:** The autoencoder is trained on reconstruction loss, which preserves variance, not trading relevance. Important but low-variance regime shifts (like a VIX backwardation spike) could get averaged away during compression.

**Solution:** Regime features bypass both the autoencoder and the FeatureExtractor. They're excluded in the dimensional reducer's `_drop_unnecessary_features`, kept as raw columns, and passed through a dedicated 2-layer MLP (1256 → 128 → 64) at the fusion layer in Actor and Critic. Only the last timestep is used (not the full window), since the regime signal is already temporally aggregated by the upstream rolling z-scores.

This means the RL loss directly shapes how regime features are compressed — the MLP learns what's useful for trading, not what's easy to reconstruct.

**Initial attempt — full window through FeatureExtractor:** The first approach was to pass the full `(B, 120, 1256)` regime window through its own FeatureExtractor (CNN + Transformer), giving 6 total FeatureExtractor instances (1 market + 1 regime per Actor, and 2 market + 2 regime in the twin Critic). This was computationally catastrophic — each regime encoder ran 3 conv layers + multi-head attention over 120 timesteps of 1,256 features, per batch, per training step. Episodes took 30+ minutes. Switching to the last-timestep MLP brought episode time back to ~11 minutes.

### GPU Oscillation Problem
After the regime separation, the FeatureExtractor only processes 128-dim market latents (fast), and the regime MLP processes a 1,256-dim vector (also fast). The GPU finished forward/backward passes faster than the CPU could prepare the next batch from the replay buffer — reconstructing `(512, 120, 128)` market windows from indices. This caused GPU utilization to oscillate between 100% and 20%.

### Solution: Batch Prefetching
A background thread prepares the next batch while the GPU processes the current one:

```python
self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
self._prefetch_future = None

# In update_parameters:
if self._prefetch_future is not None:
    batch = self._prefetch_future.result()  # already ready
else:
    batch = self.replay_buffer.sample(batch_size)

# Start preparing next batch immediately
self._prefetch_future = self._prefetch_executor.submit(
    self.replay_buffer.sample, batch_size
)
```

`reset_prefetch()` is called between episodes to drain any stale prefetched batch.

### Effect
GPU utilization became smooth. One worker thread is sufficient — the GPU consumes batches one at a time. Episode time remained ~11 minutes despite the added regime MLP — essentially zero overhead compared to the previous architecture without regime features.

---

## Fix 4: Reward Clipping

### Problem
The reward function (`portfolio_return * 100`) could produce large spikes on rare high-volatility steps, potentially destabilizing critic training.

### Solution
```python
return np.clip(portfolio_return * 100, -10.0, 10.0)
```

Caps extreme rewards while preserving relative magnitudes for the 99% of steps that fall within the range.

---

## Fix 5: Action Plot Update

### Problem
The price-with-actions plot showed markers on every step where the actor output a non-zero action. With target position, the actor outputs a target every single step, so the plot was covered in meaningless markers.

### Solution
- Track `shares_traded_at_step` (signed: positive for buys, negative for sells) alongside actions.
- Only plot markers where `shares_traded != 0` — i.e., where a trade actually executed.
- Marker alpha scales by trade magnitude relative to the largest trade in the episode.
- Buy/sell detection uses the sign of `shares_traded` (not the sign of `action`, since with target position `action > 0` doesn't necessarily mean buy).

---

## Results After All Fixes (Episode 30)

| Metric | Before (Delta Action) | After (Target Position + Alpha Floor) |
|---|---|---|
| Alpha at episode 30 | ~0.0 | 0.05 (floor) |
| Entropy at episode 30 | -0.5 (target, flatlined) | ~0.7 (still above target, decaying slowly) |
| Trade count trend | Reduced significantly as alpha collapsed | ~1,700 (still exploring fully) |
| Validation behavior | Sparse, tentative trades | Taking positions, some positive returns |
| Policy flexibility | Locked in early — shaped by first 25 episodes | Still adapting — exploration maintained |

The agent is still in early exploration (losing money on random trades), but critically:
1. Alpha is held at the floor, maintaining exploration pressure throughout training.
2. Entropy is slowly approaching the target rather than crashing to it in 25 episodes.
3. The policy will continue being shaped by episodes well beyond the initial 30, giving the agent much more time to discover profitable patterns.
4. Validation episodes show the agent taking real positions with occasional positive returns.

The true test is episodes 50-100+: whether the agent begins to reduce losses and find profitable patterns while maintaining healthy exploration.

---

## Configuration Summary

```python
target_entropy = -0.1       # conservative (more exploratory) vs default -1.0
alpha_floor = 0.05          # minimum exploration pressure
alpha_lr = 1e-4             # unchanged
actor_lr = 3e-4             # unchanged  
critic_lr = 4.5e-4          # unchanged
reward_clip = [-10, 10]     # prevent extreme reward spikes
deadband = 0.07             # 7% of max_trading_units (proportional)
max_trading_units = 30      # max shares per position
batch_size = 512            # replay buffer sample size
window_size = 120           # market data lookback
```
