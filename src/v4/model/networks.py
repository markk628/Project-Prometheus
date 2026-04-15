# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, Tuple

# from src.config.config import HIDDEN_DIM, DEVICE


# class FeatureExtractor(nn.Module):
#     """
#     Hybrid CNN-Transformer encoder for market data.
    
#     CNN branch  → captures local microstructure (spreads, ticks, 1-3 bar patterns)
#     Transformer → captures macro temporal dependencies across the full window
    
#     Both branches share an initial linear projection from raw feature space
#     (1234-dim PCA+autoencoder+time features) into a compact d_model space.
#     """

#     def __init__(
#         self,
#         feature_dim: int,          # F = 1234
#         window_size: int,          # W (number of timesteps in observation window)
#         d_model: int = 128,        # internal representation dim
#         n_heads: int = 4,
#         n_transformer_layers: int = 2,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.window_size = window_size

#         # ── shared projection: (B, W, F) → (B, W, d_model) ──────────────
#         self.proj = nn.Sequential(
#             nn.Linear(feature_dim, d_model),
#             nn.LayerNorm(d_model),
#             nn.GELU(),
#         )

#         # ── CNN branch (local patterns) ──────────────────────────────────
#         # Operates on (B, d_model, W) — channels-first for Conv1d
#         self.cnn = nn.Sequential(
#             nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(d_model * 2, d_model * 2, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.AdaptiveAvgPool1d(1),  # → (B, d_model*2, 1)
#         )
#         self.cnn_out_dim = d_model * 2  # 256 when d_model=128

#         # ── Transformer branch (global temporal dependencies) ────────────
#         self.pos_embedding = nn.Parameter(
#             torch.randn(1, window_size, d_model) * 0.02
#         )
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=n_heads,
#             dim_feedforward=d_model * 4,
#             dropout=dropout,
#             activation="gelu",
#             batch_first=True,
#             norm_first=True,       # Pre-LN (more stable training)
#         )
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=n_transformer_layers,
#             enable_nested_tensor=False
#         )
#         self.transformer_norm = nn.LayerNorm(d_model)
#         self.transformer_out_dim = d_model  # 128

#         self.out_dim = self.cnn_out_dim + self.transformer_out_dim

#     def forward(self, market_data: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             market_data: (B, W, F) — batch of windowed market observations
#         Returns:
#             (B, out_dim) — concatenation of CNN and Transformer features
#         """
#         # Handle unbatched input
#         if market_data.dim() == 2:  # (W, F)
#             market_data = market_data.unsqueeze(0)

#         x = self.proj(market_data)  # (B, W, d_model)

#         # CNN branch — (B, d_model, W)
#         cnn_in = x.permute(0, 2, 1)
#         cnn_out = self.cnn(cnn_in).squeeze(-1)  # (B, cnn_out_dim)

#         # Transformer branch
#         tf_in = x + self.pos_embedding[:, :x.size(1), :]
#         tf_out = self.transformer(tf_in)  # (B, W, d_model)
#         tf_out = self.transformer_norm(tf_out.mean(dim=1))  # mean-pool → (B, d_model)

#         return torch.cat([cnn_out, tf_out], dim=1)  # (B, out_dim)


# class Actor(nn.Module):
#     def __init__(
#         self,
#         input_shape: Tuple[int, int],    # (window_size, feature_dim)
#         portfolio_state_len: int,
#         action_dim: int = 1,
#         hidden_dim: int = HIDDEN_DIM,
#         d_model: int = 128,
#         n_heads: int = 4,
#         n_transformer_layers: int = 2,
#         dropout: float = 0.1,
#         log_std_min: float = -20.0,
#         log_std_max: float = 2.0,
#         device: torch.device = DEVICE,
#     ):
#         super().__init__()
#         self.window_size, self.feature_dim = input_shape
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.log_std_min = log_std_min
#         self.log_std_max = log_std_max
#         self.device = device

#         self.encoder = FeatureExtractor(
#             feature_dim=self.feature_dim,
#             window_size=self.window_size,
#             d_model=d_model,
#             n_heads=n_heads,
#             n_transformer_layers=n_transformer_layers,
#             dropout=dropout,
#         )

#         self.portfolio_fc = nn.Sequential(
#             nn.Linear(portfolio_state_len, hidden_dim // 4),
#             nn.GELU(),
#         )

#         fusion_dim = self.encoder.out_dim + hidden_dim // 4
#         self.trunk = nn.Sequential(
#             nn.Linear(fusion_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#         )

#         self.mean_head = nn.Linear(hidden_dim, action_dim)
#         self.log_std_head = nn.Linear(hidden_dim, action_dim)

#         self.to(device)

#     def forward(
#         self, state: Dict[str, torch.Tensor]
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         market_features = self.encoder(state["market_data"])

#         portfolio = state["portfolio_state"]
#         if portfolio.dim() == 1:
#             portfolio = portfolio.unsqueeze(0)
#         portfolio_features = self.portfolio_fc(portfolio)

#         x = torch.cat([market_features, portfolio_features], dim=1)
#         x = self.trunk(x)

#         mean = self.mean_head(x)
#         log_std = self.log_std_head(x)
#         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
#         return mean, log_std

#     def sample(
#         self, state: Dict[str, torch.Tensor]
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         mean, log_std = self.forward(state)
#         std = log_std.exp()

#         normal = torch.distributions.Normal(mean, std)
#         x_t = normal.rsample()
#         y_t = torch.tanh(x_t)

#         log_prob = normal.log_prob(x_t) - torch.log(1.0 - y_t.pow(2) + 1e-6)
#         log_prob = log_prob.sum(1, keepdim=True)

#         return y_t, log_prob, mean

#     def to(self, device: torch.device) -> "Actor":
#         self.device = device
#         return super().to(device)


# class Critic(nn.Module):
#     """Twin Q-networks, each with its own CNN-Transformer encoder."""

#     def __init__(
#         self,
#         input_shape: Tuple[int, int],
#         portfolio_state_len: int,
#         action_dim: int = 1,
#         hidden_dim: int = HIDDEN_DIM,
#         d_model: int = 128,
#         n_heads: int = 4,
#         n_transformer_layers: int = 2,
#         dropout: float = 0.1,
#         device: torch.device = DEVICE,
#     ):
#         super().__init__()
#         self.window_size, self.feature_dim = input_shape
#         self.action_dim = action_dim
#         self.hidden_dim = hidden_dim
#         self.device = device

#         # ── Q1 ───────────────────────────────────────────────────────────
#         self.q1_encoder = FeatureExtractor(
#             feature_dim=self.feature_dim,
#             window_size=self.window_size,
#             d_model=d_model,
#             n_heads=n_heads,
#             n_transformer_layers=n_transformer_layers,
#             dropout=dropout,
#         )
#         self.q1_portfolio_fc = nn.Sequential(
#             nn.Linear(portfolio_state_len, hidden_dim // 4), nn.GELU()
#         )
#         self.q1_action_fc = nn.Sequential(
#             nn.Linear(action_dim, hidden_dim // 4), nn.GELU()
#         )
#         q1_fusion_dim = self.q1_encoder.out_dim + hidden_dim // 4 + hidden_dim // 4
#         self.q1_trunk = nn.Sequential(
#             nn.Linear(q1_fusion_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, 1),
#         )

#         # ── Q2 ───────────────────────────────────────────────────────────
#         self.q2_encoder = FeatureExtractor(
#             feature_dim=self.feature_dim,
#             window_size=self.window_size,
#             d_model=d_model,
#             n_heads=n_heads,
#             n_transformer_layers=n_transformer_layers,
#             dropout=dropout,
#         )
#         self.q2_portfolio_fc = nn.Sequential(
#             nn.Linear(portfolio_state_len, hidden_dim // 4), nn.GELU()
#         )
#         self.q2_action_fc = nn.Sequential(
#             nn.Linear(action_dim, hidden_dim // 4), nn.GELU()
#         )
#         q2_fusion_dim = self.q2_encoder.out_dim + hidden_dim // 4 + hidden_dim // 4
#         self.q2_trunk = nn.Sequential(
#             nn.Linear(q2_fusion_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, 1),
#         )

#         self.to(device)

#     def forward(
#         self, state: Dict[str, torch.Tensor], action: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         portfolio = state["portfolio_state"]
#         if portfolio.dim() == 1:
#             portfolio = portfolio.unsqueeze(0)
#         if action.dim() == 1:
#             action = action.unsqueeze(0)

#         # Q1
#         q1_market = self.q1_encoder(state["market_data"])
#         q1_p = self.q1_portfolio_fc(portfolio)
#         q1_a = self.q1_action_fc(action)
#         q1 = self.q1_trunk(torch.cat([q1_market, q1_p, q1_a], dim=1))

#         # Q2
#         q2_market = self.q2_encoder(state["market_data"])
#         q2_p = self.q2_portfolio_fc(portfolio)
#         q2_a = self.q2_action_fc(action)
#         q2 = self.q2_trunk(torch.cat([q2_market, q2_p, q2_a], dim=1))

#         return q1, q2

#     def to(self, device: torch.device) -> "Critic":
#         self.device = device
#         return super().to(device)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from src.config.config import HIDDEN_DIM, DEVICE


class FeatureExtractor(nn.Module):
    """
    Hybrid CNN-Transformer encoder for market data ONLY.

    Temporal features (sine/cosine time encodings, minutes_since_open, etc.)
    are deliberately excluded — they bypass the encoder and get injected
    at the fusion layer. The encoder only sees AE latent features so it
    can focus on learning market microstructure and temporal dynamics
    from the actual market signal, not from deterministic time ramps.

    CNN branch   → raw AE latent channels into Conv1d.
                   Captures local patterns (tick microstructure, 1-5 bar).

    Transformer  → thin linear projection for head divisibility.
                   Captures global temporal dependencies across the window.
    """

    def __init__(
        self,
        feature_dim: int,              # F = 128 (AE latents only, no temporal)
        window_size: int,              # W = 120
        d_model: int = 128,            # transformer internal dim
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # ── CNN branch ───────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(feature_dim, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool1d(2),
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
            market_data: (B, W, F) — AE latent features only (no temporal)
        Returns:
            (B, out_dim) — concatenation of CNN and Transformer features
        """
        if market_data.dim() == 2:
            market_data = market_data.unsqueeze(0)

        # CNN branch
        cnn_in = market_data.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in).squeeze(-1)

        # Transformer branch
        tf_in = self.tf_proj(market_data)
        tf_in = tf_in + self.pos_embedding[:, :tf_in.size(1), :]
        tf_out = self.transformer(tf_in)
        tf_out = self.transformer_norm(tf_out.mean(dim=1))

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
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
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

        # Portfolio: 8 → 32
        self.portfolio_fc = nn.Sequential(
            nn.Linear(portfolio_state_len, hidden_dim // 4),
            nn.GELU(),
        )

        # Temporal: 8 → 32 (bypasses encoder, injected at fusion)
        self.temporal_fc = nn.Sequential(
            nn.Linear(temporal_state_len, hidden_dim // 4),
            nn.GELU(),
        )

        # Regime: 2-layer MLP on last-timestep snapshot (no sequential encoding needed)
        self.regime_fc = nn.Sequential(
            nn.Linear(regime_state_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # encoder(256) + portfolio(32) + temporal(32) + regime(64) → trunk
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
    """Twin Q-networks, each with its own CNN-Transformer encoder."""

    def __init__(
        self,
        input_shape: Tuple[int, int],
        portfolio_state_len: int,
        temporal_state_len: int,
        regime_state_len: int,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 128,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        device: torch.device = DEVICE,
    ):
        super().__init__()
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        # ── Q1 ───────────────────────────────────────────────────────────
        self.q1_encoder = FeatureExtractor(
            feature_dim=self.feature_dim,
            window_size=self.window_size,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )
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
        q1_fusion_dim = self.q1_encoder.out_dim + hidden_dim // 2 + hidden_dim // 4 * 3
        self.q1_trunk = nn.Sequential(
            nn.Linear(q1_fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ── Q2 ───────────────────────────────────────────────────────────
        self.q2_encoder = FeatureExtractor(
            feature_dim=self.feature_dim,
            window_size=self.window_size,
            d_model=d_model,
            n_heads=n_heads,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )
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
        q2_fusion_dim = self.q2_encoder.out_dim + hidden_dim // 2 + hidden_dim // 4 * 3
        self.q2_trunk = nn.Sequential(
            nn.Linear(q2_fusion_dim, hidden_dim),
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

        # Q1
        q1_market = self.q1_encoder(state["market_data"])
        q1_r = self.q1_regime_fc(regime)
        q1_p = self.q1_portfolio_fc(portfolio)
        q1_t = self.q1_temporal_fc(temporal)
        q1_a = self.q1_action_fc(action)
        q1 = self.q1_trunk(torch.cat([q1_market, q1_r, q1_p, q1_t, q1_a], dim=1))

        # Q2
        q2_market = self.q2_encoder(state["market_data"])
        q2_r = self.q2_regime_fc(regime)
        q2_p = self.q2_portfolio_fc(portfolio)
        q2_t = self.q2_temporal_fc(temporal)
        q2_a = self.q2_action_fc(action)
        q2 = self.q2_trunk(torch.cat([q2_market, q2_r, q2_p, q2_t, q2_a], dim=1))

        return q1, q2

    def to(self, device: torch.device) -> "Critic":
        self.device = device
        return super().to(device)