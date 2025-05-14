import torch
import torch.nn as nn


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        x_t = x.transpose(1, 2)
        trend = self.moving_avg(x_t)
        trend = trend.transpose(1, 2)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelationAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.proj_q(x)
        K = self.proj_k(x)
        V = self.proj_v(x)
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5), dim=-1
        )
        out = torch.bmm(attn, V)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 25, dropout: float = 0.1):
        super().__init__()
        self.decomp = SeriesDecomposition(kernel_size)
        self.attn = AutoCorrelationAttention(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        x_attn = self.attn(seasonal)
        x = self.norm1(seasonal + self.dropout(x_attn))
        x_ff = self.ff(x)
        x = self.norm2(x + self.dropout(x_ff))
        return x + trend


class TradeModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden)

        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model=hidden, dropout=dropout) for _ in range(layers)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.encoder:
            x = layer(x)
        return self.classifier(x[:, -1])
