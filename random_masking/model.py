"""Tabular Transformer for BERT-style masked reconstruction.

Architecture (faithful to BERT for tabular data):
  - Each of 12 columns (11 maskable + 1 context) becomes a token
  - For column k: token = value_proj(value) + column_embedding(k)
  - For the MASKED position, the token is REPLACED with mask_token + column_embedding(k)
    — the model never sees the actual value at the masked position
  - Self-attention runs over the 12 tokens
  - The output embedding at the masked position is projected to a scalar prediction
"""

import torch
import torch.nn as nn


class MaskedTabularTransformer(nn.Module):
    def __init__(self, n_maskable: int = 11, n_context: int = 1,
                 embed_dim: int = 32, n_heads: int = 4, n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_maskable = n_maskable
        self.n_context = n_context
        self.n_total = n_maskable + n_context
        self.embed_dim = embed_dim

        # Shared scalar-value projection (Linear(1, embed_dim) applied per column).
        self.value_proj = nn.Linear(1, embed_dim)

        # Per-column positional embedding — tells the model which slot it's seeing.
        self.column_emb = nn.Embedding(self.n_total, embed_dim)

        # Learned [MASK] token — substituted at masked positions in attention space.
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

        # Project masked-position output back to scalar.
        self.output_proj = nn.Linear(embed_dim, 1)

        # Pre-compute column indices (constant).
        self.register_buffer("col_indices", torch.arange(self.n_total))

    def forward(self, values: torch.Tensor, mask_positions: torch.Tensor) -> torch.Tensor:
        """
        values:         (batch, n_total) — all column values, standardized.
        mask_positions: (batch,)         — which column is masked per row (int).
        Returns:        (batch, 1)       — predicted value at the masked position.
        """
        batch_size = values.shape[0]
        device = values.device

        # Embed each scalar value: (batch, n_total, embed_dim).
        v = self.value_proj(values.unsqueeze(-1))

        # Column positional embeddings: (1, n_total, embed_dim).
        c = self.column_emb(self.col_indices).unsqueeze(0)

        # "Normal" tokens: value embedding + column embedding.
        value_tokens = v + c

        # "Masked" tokens: mask token + column embedding (one per column slot).
        mask_tokens = self.mask_token.unsqueeze(0).unsqueeze(0) + c  # (1, n_total, embed_dim)

        # Boolean position mask: True at the position masked for each row.
        position_mask = torch.zeros(batch_size, self.n_total, dtype=torch.bool, device=device)
        position_mask[torch.arange(batch_size, device=device), mask_positions] = True

        # Use mask_tokens at masked positions, else value_tokens. Autograd-safe.
        tokens = torch.where(
            position_mask.unsqueeze(-1),                            # (batch, n_total, 1)
            mask_tokens.expand(batch_size, -1, -1),                 # (batch, n_total, embed_dim)
            value_tokens,                                            # (batch, n_total, embed_dim)
        )

        # Self-attention over the 12 tokens.
        encoded = self.transformer(tokens)                           # (batch, n_total, embed_dim)

        # Extract the output at the masked position.
        row_idx = torch.arange(batch_size, device=device)
        masked_out = encoded[row_idx, mask_positions]                # (batch, embed_dim)

        # Project to scalar.
        return self.output_proj(masked_out)                          # (batch, 1)


def build_model(n_maskable: int = 11, n_context: int = 1) -> nn.Module:
    return MaskedTabularTransformer(n_maskable=n_maskable, n_context=n_context)