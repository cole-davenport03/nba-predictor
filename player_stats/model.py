"""Tabular Transformer for predicting PTS from box-score features.

Each of the 11 input features becomes a token. Self-attention lets the model
learn relationships between features. A learned [CLS] token aggregates the
result for the final scalar prediction.

Same input/output contract as the MLP it replaces:
   forward(x): (batch, n_features) -> (batch, 1)
"""

import torch
import torch.nn as nn


class TabularTransformer(nn.Module):
    def __init__(self, n_features: int = 11, embed_dim: int = 32,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim

        # Project each scalar feature value to an embedding vector.
        self.value_proj = nn.Linear(1, embed_dim)

        # Learned positional embedding per feature column —
        # tells the model "this is the MIN slot vs. FG_PCT slot vs. ..."
        self.column_emb = nn.Embedding(n_features, embed_dim)

        # Learned [CLS] token whose final hidden state we use as the prediction.
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

        # Pre-compute column indices (won't change during training).
        self.register_buffer("col_indices", torch.arange(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        batch_size = x.shape[0]

        # Embed each scalar value to embed_dim, then add the column position embedding.
        v = self.value_proj(x.unsqueeze(-1))               # (batch, n_features, embed_dim)
        c = self.column_emb(self.col_indices).unsqueeze(0) # (1, n_features, embed_dim)
        tokens = v + c                                      # (batch, n_features, embed_dim)

        # Prepend CLS token.
        cls = self.cls_token.expand(batch_size, -1, -1)    # (batch, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)            # (batch, n_features+1, embed_dim)

        # Pass through transformer encoder.
        encoded = self.transformer(tokens)                  # (batch, n_features+1, embed_dim)

        # Extract CLS output and project to scalar.
        cls_out = encoded[:, 0, :]                          # (batch, embed_dim)
        return self.output_proj(cls_out)                    # (batch, 1)


def build_model(n_features: int = 11) -> nn.Module:
    return TabularTransformer(n_features=n_features)