# Dev Log: Alpha Floor Adjustment

## Date: April 10, 2026

---

## Problem: Alpha Floor Too High

### Observation (Episode 140)
After implementing the alpha floor of 0.05, entropy stabilized at ~0.667 — far above the target of -0.1. Entropy dropped from 0.683 to 0.665 over the first 60 episodes, then reversed and climbed back to ~0.669 where it flatlined for the remaining 80 episodes. This indicated that the floor created an equilibrium where the 0.05 entropy bonus in the actor loss perfectly balanced the Q-value gradient, preventing the policy from ever becoming selective.

### Evidence
- **Entropy**: Flatlined at ~0.667 from episode 60 onward. Needed to reach -0.1 — a gap of 0.77 with no downward movement.
- **Train returns**: 10-episode MA oscillated around -20% with no trend across 140 episodes.
- **Trade count**: Constant at ~1,700 trades per episode — pure random churning with no reduction.
- **Win rate**: Flat at ~25% — consistent with random trading.
- **Validation**: Performed reasonably (buy-and-hold) because validation uses deterministic policy (no entropy), confirming the networks were learning something but the training exploration noise was too high to refine it.

### Diagnosis
The alpha floor of 0.05 over-corrected the original alpha collapse problem. The original issue was alpha reaching 0.0, causing entropy to freefall. The floor was meant to be a safety net, but 0.05 was high enough to act as a ceiling on policy refinement instead.

---

## Fix: Lower Alpha Floor from 0.05 to 0.01

```python
# In update_parameters:
self.alpha = torch.clamp(self.log_alpha.exp(), min=0.01)

# In load_model:
self.alpha = torch.clamp(self.log_alpha.exp(), min=0.01)
```

### Rationale
- 0.01 still prevents the complete collapse to zero that killed earlier runs.
- The entropy bonus `0.01 * log_probs` is small enough that Q-value gradients dominate, allowing entropy to actually descend toward -0.1.
- The policy can become selective while maintaining a minimal exploration floor.

### Expected Outcome
- Entropy should descend toward -0.1 rather than plateauing at 0.667.
- Train trade count should decrease from ~1,700 as the policy becomes more selective.
- Train returns should begin improving as random churning is reduced.
- Validation should remain stable or improve.
