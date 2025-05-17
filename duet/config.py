DEFAULT_TRANSFORMER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "hidden_size": 256,
    "factor": 1,
    "activation": "gelu",
    "moving_avg": 25,
    "dropout": 0.2,
    "fc_dropout": 0.1,
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "groups": [[0], [1, 2], [3]],
}


class DUETConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
