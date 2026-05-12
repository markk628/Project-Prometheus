# ─────────────────────────────────────────────────────────────────────────────
# TODO — architecture experiments worth trying once a baseline run exists
#
# The current FeatureExtractor processes a (60, 25) market-data sequence, but
# many of those 25 features are already multi-horizon summaries themselves —
# log_return_5/20/60, volatility_5_20_ratio, ema_5_20_ratio, adx_5_20_ratio,
# volume_5_20_ratio, and so on. Feeding 60 timesteps of multi-horizon
# summaries through a sequence encoder is probably redundant with what the
# features already encode.
#
# Two options to try once a baseline is established:
#
# 1) MLP-only market path (most aggressive).
#    Drop the encoder. Pass the last-timestep 25-dim market vector through an
#    MLP, the same way regime features are handled today.
#
#    PREREQUISITE — add delta features in feature_engineer.py first.
#    The regime MLP works on a last-timestep snapshot because its features
#    already include explicit deltas (breadth_*_delta_1/5/20,
#    vix_term_*_delta_1/5/20), so the snapshot carries trajectory info. The
#    25 ticker features do NOT have this — they're levels and ratios, not
#    changes. An MLP seeing `adx_14 = 25` can't distinguish "building for
#    30 days" from "spiked yesterday". For the symmetry to hold, add e.g.
#    volatility_5_20_ratio_delta_5, adx_14_delta_5, ema_close_ratio_20_delta_5
#    before removing the encoder. Without deltas, this option strictly
#    reduces the information reaching the policy vs. the current encoder.
#
# 2) Minimal CNN + MLP (middle ground, cheap insurance).
#    Single 1D conv layer (kernel=3, ~32 channels) + AdaptiveAvgPool -> 32-dim,
#    no transformer. Captures local cross-feature patterns (e.g. volatility
#    rising while volume-price corr flips sign) without the cost of the full
#    hybrid. Roughly ~3k encoder params vs ~90k for the current simplified
#    version. No feature-engineering changes needed.
#
# Empirical question which lands where. Run the current architecture as
# baseline; then compare. If option 1 (with deltas added) matches baseline
# validation, simplicity wins. If there's a clear gap, the CNN is earning
# its keep.
# ─────────────────────────────────────────────────────────────────────────────


import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.config.config import HIDDEN_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    """
    Hybrid CNN + Transformer encoder for daily market data.

    Operates on (B, W, F) where W is the lookback window in trading days
    (typically 60) and F is the number of per-ticker market features
    (~45: returns, volatility, trend, volume, candlestick).

    CNN branch   → 2 conv layers over the time axis. Captures local
                   3-5 day patterns (short-term momentum, micro-reversals).
    Transformer  → 1 encoder layer with positional embedding. Captures
                   longer-range dependencies across the full window.

    Temporal (sin/cos) and regime features bypass this encoder and enter
    the fusion layer directly — they're already compact and don't benefit
    from sequential modeling at this timescale.
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
        self.d_model = d_model
        self.window_size = window_size

        # ── CNN branch ───────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(feature_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cnn_out_dim = d_model

        # ── Transformer branch ───────────────────────────────────────────
        self.tf_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, window_size, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
            enable_nested_tensor=False,
        )
        self.transformer_norm = nn.LayerNorm(d_model)
        self.transformer_out_dim = d_model

        self.out_dim = self.cnn_out_dim + self.transformer_out_dim

    def forward(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            market_data: (B, W, F)
        Returns:
            (B, out_dim) — concat of CNN + Transformer features
        """
        if market_data.dim() == 2:
            market_data = market_data.unsqueeze(0)

        # CNN branch
        cnn_in = market_data.permute(0, 2, 1)              # (B, F, W)
        cnn_out = self.cnn(cnn_in).squeeze(-1)             # (B, d_model)

        # Transformer branch
        tf_in = self.tf_proj(market_data)
        tf_in = tf_in + self.pos_embedding[:, :tf_in.size(1), :]
        tf_out = self.transformer(tf_in)
        tf_out = self.transformer_norm(tf_out.mean(dim=1))  # (B, d_model)

        return torch.cat([cnn_out, tf_out], dim=1)


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