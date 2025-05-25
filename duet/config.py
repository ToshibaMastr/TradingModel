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
    fc_dropout: float = 0.1

    moving_avg: int = 25

    gate_size: int = 256
    num_experts: int = 4
    noisy_gating: int = True
    k: int = 1
