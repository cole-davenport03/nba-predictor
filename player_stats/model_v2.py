"""Player-aware tabular transformer for PTS regression.

Extends the v1 TabularTransformer in model.py by adding a learned PLAYER
embedding that's added to every column token. The model now knows
"this row belongs to SGA" alongside the box-score values, which lets it
learn player-specific scoring tendencies — a Luka game with 10 AST and 40 MIN
predicts very differently from a Dort game with the same inputs.

Architecture:
    token_k = value_proj(value_k) + column_emb(k) + player_emb(player_id)
    CLS     = cls_token + player_emb(player_id)
    self-attention over [CLS, token_0, ..., token_{F-1}]
    output  = linear(CLS_out)
"""

import torch
import torch.nn as nn


class PlayerAwareTransformer(nn.Module):
    def __init__(self, n_features: int = 11, n_players: int = 46,
                 embed_dim: int = 32, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.n_players = n_players
        self.embed_dim = embed_dim

        # Scalar-value → embed_dim projection.
        self.value_proj = nn.Linear(1, embed_dim)

        # Per-column positional embedding (which slot — MIN vs FG_PCT vs ...).
        self.column_emb = nn.Embedding(n_features, embed_dim)

        # Per-player identity embedding (the key new piece).
        self.player_emb = nn.Embedding(n_players, embed_dim)

        # Learned [CLS] token whose final state we read out.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, 1)

        self.register_buffer("col_indices", torch.arange(n_features))

    def forward(self, x: torch.Tensor, player_ids: torch.Tensor) -> torch.Tensor:
        """
        x:          (batch, n_features) standardized feature values
        player_ids: (batch,)             integer player IDs
        returns:    (batch, 1)           predicted standardized PTS
        """
        batch_size = x.shape[0]

        # Per-token components.
        v = self.value_proj(x.unsqueeze(-1))                  # (B, F, E)
        c = self.column_emb(self.col_indices).unsqueeze(0)    # (1, F, E) — broadcasts
        p = self.player_emb(player_ids).unsqueeze(1)          # (B, 1, E) — broadcasts over columns

        tokens = v + c + p                                     # (B, F, E)

        # Prepend a per-player-biased CLS token.
        cls = self.cls_token.expand(batch_size, -1, -1) + p   # (B, 1, E)
        tokens = torch.cat([cls, tokens], dim=1)              # (B, F+1, E)

        encoded = self.transformer(tokens)                    # (B, F+1, E)
        cls_out = encoded[:, 0, :]                            # (B, E)
        return self.output_proj(cls_out)                      # (B, 1)


def build_model(n_features: int = 11, n_players: int = 46) -> nn.Module:
    return PlayerAwareTransformer(n_features=n_features, n_players=n_players)
