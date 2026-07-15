# ─────────────────────────────────────────────────────────────────────────────
# v6 run 4b: MLP-only market path
#
# The 4a run added per-ticker delta features (ema_close_ratio_20_delta_20,
# ema_20_60_ratio_delta_20, adx_5_20_ratio_delta_20, volatility_5_20_ratio_
# delta_20, volume_5_20_ratio_delta_20) inside _normalize_data. Result was
# neutral vs v5 baseline — encoder tolerated the additions without much
# benefit, consistent with the encoder using temporal info already (or
# alternatively with the encoder mostly ignoring it). 4b tests the
# discriminating question: does the encoder actually need to see the
# 60-day window, or is a last-timestep snapshot enough?
#
# Implementation: replace the CNN+Transformer body with a 3-layer MLP that
# operates on market_data[:, -1, :] (the most recent bar). Keep the same
# input contract (B, W, F) so the env / replay buffer / trainer don't need
# changes — the window is computed and stored as before, the network just
# ignores all timesteps except the last. Output dim stays at 128 (== the
# encoder's 2 * d_model with default args) so the downstream Actor/Critic
# fusion path is byte-identical to 4a's.
#
# Decision rule (pre-committed before training):
#   - Cross-seed mean within ~1pt return / ~0.03 Sharpe of the 4a baseline
#     on aggregate -> MLP passes, ships as the v6 capstone architecture.
#   - Underperform by more than that -> encoder is doing real temporal work
#     and stays in. Revert to 4a by git revert of this commit.
#
# The "win condition" here is matching the encoder, not beating it. The
# value of MLP-passing is architectural simplification: ~40% the encoder's
# parameter count, faster training, simpler codebase, and a shape closer
# to what v7's portfolio-allocation problem will likely want.
# ─────────────────────────────────────────────────────────────────────────────


import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.config.config import HIDDEN_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    """
    MLP-only market encoder for daily market data (v6 run 4b).

    Takes a (B, W, F) market-data window for API compatibility with the
    rest of the pipeline, but only uses the most recent bar (the last
    timestep along the time axis). The per-ticker delta features added
    in run 4a give the snapshot the trajectory info that an MLP would
    otherwise lack.

    The constructor accepts the same arguments as the previous
    CNN+Transformer version so callers (Actor, Critic) need no changes.
    The unused arguments (`n_heads`, `n_transformer_layers`, `dropout`)
    are kept in the signature for backwards compatibility with the call
    sites; only `feature_dim` and `d_model` actually influence behavior.

    Architecture:
        last-timestep slice (B, F)
            → Linear(F, 128) + LayerNorm + GELU
            → Linear(128, 128) + LayerNorm + GELU
            → Linear(128, 128) + LayerNorm + GELU
        → (B, 128)

    Output dim is fixed at `2 * d_model` to match the previous encoder's
    contract (CNN d_model + Transformer d_model concatenated).
    """

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # window_size, n_heads, n_transformer_layers, dropout retained in
        # signature for caller compatibility but unused. The MLP only sees
        # one timestep regardless of window_size.
        del n_heads, n_transformer_layers, dropout

        self.window_size = window_size
        self.d_model = d_model
        # Match the previous encoder's out_dim (CNN d_model + Transformer
        # d_model) so the Actor/Critic fusion math doesn't change.
        self.out_dim = 2 * d_model

        hidden = self.out_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

    def forward(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_data: (B, W, F). Only the last timestep along W is used.
        Returns:
            (B, out_dim) — MLP encoding of the most recent bar's features.
        """
        if market_data.dim() == 2:
            market_data = market_data.unsqueeze(0)

        # Take the most recent bar. Per-ticker deltas added in 4a carry
        # the trajectory info the rest of the window would have provided.
        last_bar = market_data[:, -1, :]                # (B, F)
        return self.mlp(last_bar)                       # (B, out_dim)


class Actor(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],        # (window_size, market_feature_dim)
        portfolio_state_len: int,
        temporal_state_len: int,
        regime_state_len: int,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 1,
        dropout: float = 0.1,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        self.encoder = FeatureExtractor(
            feature_dim=self.feature_dim,
            window_size=self.window_size,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )

        # Portfolio → hidden_dim // 4
        self.portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_state_len, hidden_dim // 4),
            nn.GELU(),
        )

        # Temporal → hidden_dim // 4 (bypasses encoder, injected at fusion)
        self.temporal_fc = nn.Sequential(
            nn.Linear(temporal_state_len, hidden_dim // 4),
            nn.GELU(),
        )

        # Regime: 2-layer MLP on last-timestep snapshot
        self.regime_fc = nn.Sequential(
            nn.Linear(regime_state_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # encoder(2*d_model) + portfolio(H/4) + temporal(H/4) + regime(H/2)
        fusion_dim = self.encoder.out_dim + hidden_dim // 4 + hidden_dim // 4 + hidden_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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

        x = torch.cat([market_features, portfolio_features, temporal_features, regime_features], dim=1)
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

        log_prob = normal.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return y_t, log_prob, mean

    def to(self, device: torch.device) -> "Actor":
        self.device = device
        return super().to(device)


class Critic(nn.Module):
    """
    Twin Q-networks with a SHARED market encoder.

    Both Q heads receive the same market representation from a single
    FeatureExtractor but keep their own regime/portfolio/temporal/action
    MLPs and trunks. The twin-Q overestimation-bias correction still
    works because the two heads have different weights and training
    trajectories — they simply share the expensive sequential encoder.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        portfolio_state_len: int,
        temporal_state_len: int,
        regime_state_len: int,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_transformer_layers: int = 1,
        dropout: float = 0.1,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # ── Shared market encoder ────────────────────────────────────────
        self.encoder = FeatureExtractor(
            feature_dim=self.feature_dim,
            window_size=self.window_size,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )
        fusion_dim = self.encoder.out_dim + hidden_dim // 2 + (hidden_dim // 4) * 3

        # ── Q1 head ──────────────────────────────────────────────────────
        self.q1_regime_fc = nn.Sequential(
            nn.Linear(regime_state_len, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.q1_portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_state_len, hidden_dim // 4), nn.GELU()
        )
        self.q1_temporal_fc = nn.Sequential(
            nn.Linear(temporal_state_len, hidden_dim // 4), nn.GELU()
        )
        self.q1_action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4), nn.GELU()
        )
        self.q1_trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ── Q2 head ──────────────────────────────────────────────────────
        self.q2_regime_fc = nn.Sequential(
            nn.Linear(regime_state_len, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        self.q2_portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_state_len, hidden_dim // 4), nn.GELU()
        )
        self.q2_temporal_fc = nn.Sequential(
            nn.Linear(temporal_state_len, hidden_dim // 4), nn.GELU()
        )
        self.q2_action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4), nn.GELU()
        )
        self.q2_trunk = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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

        # Shared encoder pass (computed once, used by both Q heads)
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