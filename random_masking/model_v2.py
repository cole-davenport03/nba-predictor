"""Player-aware BERT-style transformer for masked-column reconstruction.

Extends the v1 MaskedTabularTransformer in model.py by adding a learned
PLAYER embedding that's added to every token — both value tokens and the
[MASK] token. The model now sees "this row belongs to SGA" while it
reconstructs the masked column, giving it a strong player-specific prior.

Forward signature gains a `player_ids` arg; everything else is identical
to v1.
"""

import torch
import torch.nn as nn


class MaskedPlayerAwareTransformer(nn.Module):
    def __init__(self, n_maskable: int = 11, n_context: int = 1,
                 n_players: int = 46, embed_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_maskable = n_maskable
        self.n_context = n_context
        self.n_total = n_maskable + n_context
        self.n_players = n_players
        self.embed_dim = embed_dim

        self.value_proj = nn.Linear(1, embed_dim)
        self.column_emb = nn.Embedding(self.n_total, embed_dim)
        self.player_emb = nn.Embedding(n_players, embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

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

        self.register_buffer("col_indices", torch.arange(self.n_total))

    def forward(self, values: torch.Tensor, mask_positions: torch.Tensor,
                player_ids: torch.Tensor) -> torch.Tensor:
        """
        values:         (batch, n_total)  — standardized column values
        mask_positions: (batch,)          — which column is masked per row
        player_ids:     (batch,)          — integer player IDs
        returns:        (batch, 1)        — predicted value at the masked position
        """
        batch_size = values.shape[0]
        device = values.device

        v = self.value_proj(values.unsqueeze(-1))                       # (B, N, E)
        c = self.column_emb(self.col_indices).unsqueeze(0)              # (1, N, E)
        p = self.player_emb(player_ids).unsqueeze(1)                    # (B, 1, E)

        # "Normal" tokens: value + column + player.
        value_tokens = v + c + p                                         # (B, N, E)

        # "Mask" tokens: mask_token + column + player. (Same player tag, no value.)
        mask_tokens = (self.mask_token.unsqueeze(0).unsqueeze(0) + c + p)  # (B, N, E)

        position_mask = torch.zeros(batch_size, self.n_total, dtype=torch.bool, device=device)
        position_mask[torch.arange(batch_size, device=device), mask_positions] = True

        tokens = torch.where(
            position_mask.unsqueeze(-1),
            mask_tokens,
            value_tokens,
        )

        encoded = self.transformer(tokens)                              # (B, N, E)
        row_idx = torch.arange(batch_size, device=device)
        masked_out = encoded[row_idx, mask_positions]                   # (B, E)
        return self.output_proj(masked_out)                             # (B, 1)


def build_model(n_maskable: int = 11, n_context: int = 1,
                n_players: int = 46) -> nn.Module:
    return MaskedPlayerAwareTransformer(
        n_maskable=n_maskable, n_context=n_context, n_players=n_players,
    )
