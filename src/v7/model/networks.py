# ─────────────────────────────────────────────────────────────────────────────
# v7: Multi-ticker allocation networks
#
# Architecture:
#
# 1. FeatureExtractor is MLP-only on the current-bar features per ticker.
#    Takes 3D market_data (B, N, F_per_ticker) — no window dimension.
#    The shared MLP processes each of the N tickers' features through the
#    same weights, then the N outputs are concatenated. This gives the
#    policy a per-ticker representation that fusion can then specialize on
#    (action[i] specializes to ticker[i]).
#
#    v7 committed to MLP-on-last-bar based on v6 run 4b's evidence: the
#    encoder didn't actually use the 60-bar window — per-ticker delta
#    features added in 4a/4c carry the trajectory info a window would
#    otherwise provide. v6 left the window plumbing in place because
#    removing it was more invasive than the single-ablation result
#    warranted; v7's commitment to this architecture made it worth
#    stripping out the dead weight.
#
# 2. Per-ticker output dim is d_model (=64 by default). With N=5 tickers
#    this gives encoder.out_dim = n_tickers * d_model = 320.
#
# 3. Actor/Critic action_dim defaults to 5 (basket size). The action is
#    interpreted by the env via softmax-with-fixed-cash-logit; the network
#    itself just outputs 5-dim tanh-squashed continuous values, no
#    behavioral change to SAC's sampling math.
#
# 4. Portfolio state width grows from 7 to 22 (caller passes
#    portfolio_state_len; constructor doesn't hardcode it). portfolio_fc
#    structure is unchanged — same 22→32 MLP shape as v6's 7→32.
#
# Temporal (6) and regime (~45) paths are unchanged from v6.
# ─────────────────────────────────────────────────────────────────────────────


import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.config.config import HIDDEN_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    """
    v7 shared multi-ticker market encoder.

    Forward input:  market_data shape (B, N, F_per_ticker).
    Forward output: (B, N * d_model). The N dim is preserved in the output
    layout so a downstream fusion layer can implicitly attribute features
    to specific tickers (concat order matches V7_BASKET order).

    Mechanism: the same MLP processes every ticker's current-bar features.
    Conceptually "given a per-ticker state snapshot, here's the relevant
    representation" — universal across the basket. Per-ticker specialization
    happens later, in the trunk/heads, which see N specific positions in
    the concatenated representation.

    Unused args (`n_heads`, `n_transformer_layers`, `dropout`) are kept in
    the signature for backwards compat with v6 call sites; only
    `feature_dim`, `n_tickers`, and `d_model` actually influence behavior.
    """

    def __init__(
        self,
        feature_dim: int,        # per-ticker feature width
        n_tickers: int,          # N (= 5 for default V7_BASKET)
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        del n_heads, n_transformer_layers, dropout

        if n_tickers < 1:
            raise ValueError(f"n_tickers must be ≥ 1, got {n_tickers}")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be ≥ 1, got {feature_dim}")

        self.feature_dim = feature_dim
        self.n_tickers = n_tickers
        self.d_model = d_model
        self.out_dim = n_tickers * d_model

        # Shared per-ticker MLP. Internal hidden = 2*d_model so capacity is
        # comparable to v6's 4b encoder; output projected to d_model for a
        # compact per-ticker representation.
        hidden = 2 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_data: (B, N, F_per_ticker), or (N, F_per_ticker) for
                single-sample inference (a leading batch dim is added).
        Returns:
            (B, N * d_model) — per-ticker encodings concatenated in ticker order.
        """
        if market_data.dim() == 2:
            # (N, F) → add batch dim. Convenient for one-off non-batched eval.
            market_data = market_data.unsqueeze(0)
        if market_data.dim() != 3:
            raise ValueError(
                f"FeatureExtractor expects (B, N, F), got shape {tuple(market_data.shape)}"
            )

        B, N, F = market_data.shape
        if N != self.n_tickers:
            raise ValueError(f"Got N={N} tickers, expected {self.n_tickers}")
        if F != self.feature_dim:
            raise ValueError(f"Got F={F} features, expected {self.feature_dim}")

        # Apply shared MLP per ticker: (B*N, F) → (B*N, d_model)
        flat = market_data.reshape(B * N, F)         # (B*N, F)
        encoded = self.mlp(flat)                     # (B*N, d_model)

        # Restore the batch/ticker structure and flatten N into the feature dim.
        # The final (B, N*d_model) preserves per-ticker order — fusion sees
        # ticker 0 in dims [0:d_model], ticker 1 in [d_model:2*d_model], etc.
        return encoded.reshape(B, N * self.d_model)  # (B, N * d_model)


class Actor(nn.Module):
    """
    v7 multi-ticker SAC actor.

    Outputs an action_dim-dim (default 5) tanh-squashed Gaussian. The env
    interprets action via softmax-with-fixed-cash-logit, producing
    (action_dim + 1) target weights summing to 1.

    State dict expected from env:
        market_data:      (B, N, F_per_ticker)
        portfolio_state:  (B, portfolio_state_len)   — 22 for v7 default
        temporal:         (B, temporal_state_len)    — 6 for v7
        regime:           (B, regime_state_len)      — ~45 for v7
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],   # (n_tickers, feature_dim_per_ticker)
        portfolio_state_len: int,
        temporal_state_len: int,
        regime_state_len: int,
        action_dim: int = 5,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 1,
        dropout: float = 0.1,
        log_std_min: float = -20.0,
        # v7: log_std_max tightened from 2.0 → 0.0. v6 used log_std_max=2
        # (max per-dim std ≈ 7.4) with 1-dim action; in 5-dim action space
        # that produces extremely wide joint exploration (Gaussian in R⁵
        # with std 7.4 per axis covers an enormous volume) and contributes
        # to extreme x_t values pre-tanh, which feeds the SAC numerical
        # instability path. log_std_max=0 caps per-dim std at 1.0 — still
        # plenty of exploration in tanh-squashed action space, much
        # better-behaved numerics.
        log_std_max: float = 0.0,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(
                f"v7 input_shape must be (N, F_per_ticker); got {input_shape}"
            )
        self.n_tickers, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        self.encoder = FeatureExtractor(
            feature_dim=self.feature_dim,
            n_tickers=self.n_tickers,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )

        # Portfolio → hidden_dim // 4. Width unchanged from v6; only the
        # input dim (portfolio_state_len) grew, from 7 → 22.
        self.portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_state_len, hidden_dim // 4),
            nn.GELU(),
        )

        # Temporal → hidden_dim // 4. Identical to v6.
        self.temporal_fc = nn.Sequential(
            nn.Linear(temporal_state_len, hidden_dim // 4),
            nn.GELU(),
        )

        # Regime: 2-layer MLP. Identical to v6.
        self.regime_fc = nn.Sequential(
            nn.Linear(regime_state_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # encoder(N*d_model) + portfolio(H/4) + temporal(H/4) + regime(H/2)
        fusion_dim = (
            self.encoder.out_dim
            + hidden_dim // 4
            + hidden_dim // 4
            + hidden_dim // 2
        )
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            # v7 run 2: dropout for regularization. Applied between
            # Linear+GELU blocks in the trunk (where most params live).
            # Skipped in the small input projections and per-ticker
            # encoder MLP (already shallow). dropout=0 disables.
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

        self.to(device)

    def forward(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        market_features = self.encoder(state["market_data"])

        portfolio = state["portfolio_state"]
        if portfolio.dim() == 1:
            portfolio = portfolio.unsqueeze(0)
        portfolio_features = self.portfolio_fc(portfolio)

        temporal = state["temporal"]
        if temporal.dim() == 1:
            temporal = temporal.unsqueeze(0)
        temporal_features = self.temporal_fc(temporal)

        regime = state["regime"]
        if regime.dim() == 1:
            regime = regime.unsqueeze(0)
        regime_features = self.regime_fc(regime)

        x = torch.cat(
            [market_features, portfolio_features, temporal_features, regime_features],
            dim=1,
        )
        x = self.trunk(x)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # tanh correction term. eps bumped from 1e-6 to 1e-4 for v7's 5-dim
        # action: at 1e-6, each dim's correction caps at log(1e-6) ≈ -13.8,
        # summed across 5 dims that's worst-case -69. Combined with the SAC
        # target-Q formula `Q_target = r + γ(Q_next - α * log_prob)`, this
        # could produce target magnitudes that overflow float32 during MSE
        # squaring, causing NaN gradients and the NaN-cascade that took out
        # the run-1 dry-run extension at ep 55. 1e-4 caps each dim at -9.2,
        # sum across 5 dims at -46, well inside float32's safe range.
        # See dev_log_v7.md (run-1 NaN incident) for details.
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-4)
        log_prob = log_prob.sum(1, keepdim=True)

        return y_t, log_prob, mean

    def to(self, device: torch.device) -> "Actor":
        self.device = device
        return super().to(device)


class Critic(nn.Module):
    """
    v7 twin Q-networks with a SHARED multi-ticker market encoder.

    Both Q heads use the same FeatureExtractor for the expensive per-ticker
    encoding, but keep their own portfolio/temporal/regime/action MLPs and
    trunks. Twin-Q overestimation-bias correction still works since the heads
    have independent weights and training trajectories.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],   # (n_tickers, feature_dim_per_ticker)
        portfolio_state_len: int,
        temporal_state_len: int,
        regime_state_len: int,
        action_dim: int = 5,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 1,
        dropout: float = 0.1,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(
                f"v7 input_shape must be (N, F_per_ticker); got {input_shape}"
            )
        self.n_tickers, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # ── Shared market encoder (single instance, used by both Q heads) ──
        self.encoder = FeatureExtractor(
            feature_dim=self.feature_dim,
            n_tickers=self.n_tickers,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )

        # encoder(N*d_model) + regime(H/2) + portfolio(H/4) + temporal(H/4) + action(H/4)
        fusion_dim = (
            self.encoder.out_dim
            + hidden_dim // 2
            + (hidden_dim // 4) * 3
        )

        # ── Q1 head ──
        self.q1_regime_fc = nn.Sequential(
            nn.Linear(regime_state_len, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.q1_portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_state_len, hidden_dim // 4), nn.GELU(),
        )
        self.q1_temporal_fc = nn.Sequential(
            nn.Linear(temporal_state_len, hidden_dim // 4), nn.GELU(),
        )
        self.q1_action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4), nn.GELU(),
        )
        self.q1_trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),   # v7 run 2: regularization, see Actor trunk
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        # ── Q2 head ──
        self.q2_regime_fc = nn.Sequential(
            nn.Linear(regime_state_len, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.q2_portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_state_len, hidden_dim // 4), nn.GELU(),
        )
        self.q2_temporal_fc = nn.Sequential(
            nn.Linear(temporal_state_len, hidden_dim // 4), nn.GELU(),
        )
        self.q2_action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4), nn.GELU(),
        )
        self.q2_trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.to(device)

    def forward(
        self, state: Dict[str, torch.Tensor], action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        portfolio = state["portfolio_state"]
        if portfolio.dim() == 1:
            portfolio = portfolio.unsqueeze(0)

        temporal = state["temporal"]
        if temporal.dim() == 1:
            temporal = temporal.unsqueeze(0)

        regime = state["regime"]
        if regime.dim() == 1:
            regime = regime.unsqueeze(0)

        if action.dim() == 1:
            action = action.unsqueeze(0)

        # Shared encoder pass (computed once, used by both Q heads).
        market_features = self.encoder(state["market_data"])

        # Q1
        q1_r = self.q1_regime_fc(regime)
        q1_p = self.q1_portfolio_fc(portfolio)
        q1_t = self.q1_temporal_fc(temporal)
        q1_a = self.q1_action_fc(action)
        q1 = self.q1_trunk(torch.cat([market_features, q1_r, q1_p, q1_t, q1_a], dim=1))

        # Q2
        q2_r = self.q2_regime_fc(regime)
        q2_p = self.q2_portfolio_fc(portfolio)
        q2_t = self.q2_temporal_fc(temporal)
        q2_a = self.q2_action_fc(action)
        q2 = self.q2_trunk(torch.cat([market_features, q2_r, q2_p, q2_t, q2_a], dim=1))

        return q1, q2

    def to(self, device: torch.device) -> "Critic":
        self.device = device
        return super().to(device)