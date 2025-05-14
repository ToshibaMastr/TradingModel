from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, input: int, hidden: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.residual = hidden == input

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        out = self.dropout(out)
        if self.residual:
            out += x
        return out


class TradeModel(nn.Module):
    def __init__(
        self, input, output, hidden: int = 128, layers: int = 1, dropout: float = 0.2
    ):
        super().__init__()

        self.lstm_blocks = nn.ModuleList()
        self.lstm_blocks.append(ResidualBlock(input, hidden, dropout))

        for _ in range(1, layers):
            self.lstm_blocks.append(ResidualBlock(hidden, hidden, dropout))

        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, output),
        )

    def forward(self, x):
        for block in self.lstm_blocks:
            x = block(x)
        x = self.classifier(x[:, -1, :])
        return x
