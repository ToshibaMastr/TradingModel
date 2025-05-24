from torch import nn


class TradeModel(nn.Module):
    def __init__(self, tokens=3, frame=2, hidden=64, layers=1, dropout=0.1):
        super().__init__()

        self.token_emb = nn.Embedding(tokens, hidden)
        self.frame_proj = nn.Linear(frame, hidden)

        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )

        self.policy_head = nn.Linear(hidden, tokens)

    def forward(self, token_seq, frame_seq, hc=None):
        tok_emb = self.token_emb(token_seq)
        frm_emb = self.frame_proj(frame_seq)

        h, hc = self.gru(tok_emb + frm_emb, hc)
        logits = self.policy_head(h)

        return logits, hc
