import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return residual + x


class TabResNetAttn(nn.Module):
    """
    Tabular self-attention residual model.

    This implementation expects preprocessed numeric input features.
    Each input feature is projected into a feature token.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        token_dim: int = 16,
        num_heads: int = 4,
        hidden_dim: int = 128,
        num_residual_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if token_dim % num_heads != 0:
            raise ValueError(
                "token_dim must be divisible by num_heads"
            )

        self.num_features = num_features
        self.token_dim = token_dim

        self.feature_projection = nn.ModuleList([
            nn.Linear(1, token_dim)
            for _ in range(num_features)
        ])

        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(token_dim)

        self.projection = nn.Linear(
            num_features * token_dim,
            hidden_dim,
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_residual_blocks)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x shape: [batch_size, num_features]
        tokens = []

        for i, projection in enumerate(self.feature_projection):
            feature = x[:, i:i + 1]
            token = projection(feature)
            tokens.append(token)

        # [batch_size, num_features, token_dim]
        tokens = torch.stack(tokens, dim=1)

        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        tokens = self.attention_norm(tokens + attended)

        # Flatten feature tokens
        x = tokens.reshape(tokens.shape[0], -1)

        x = self.projection(x)
        x = self.blocks(x)

        return self.head(x)
