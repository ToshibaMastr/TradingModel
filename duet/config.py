class DUETConfig:
    seq_len: int = 256
    pred_len: int = 24

    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1

    d_model: int = 512
    d_ff: int = 2048

    n_heads: int = 8
    e_layers: int = 2
    factor: int = 1

    activation: str = "gelu"
    dropout: float = 0.2
    fc_dropout: float = 0.05

    moving_avg: int = 25

    gate_size: int = 128
    num_experts: int = 4
    noisy_gating: int = True
    k: int = 3

    def name(self) -> str:
        return (
            f"S{self.seq_len}P{self.pred_len}D{self.d_model}"
            f"F{self.d_ff}H{self.n_heads}E{self.e_layers}G{self.gate_size}"
            f"N{self.num_experts}K{self.k}O{int(self.noisy_gating)}"
            f"R{self.dropout}C{self.fc_dropout}A{self.activation}"
            f"M{self.moving_avg}".replace(".", "-")
        )
