DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "d_model": 512,
    "d_ff": 2048,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "hidden_size": 256,
    "freq": "h",
    "factor": 1,
    "seg_len": 6,
    "activation": "gelu",
    "output_attention": 0,
    "dropout": 0.2,
    "fc_dropout": 0.2,
    "moving_avg": 25,
    "lradj": "type3",
    "loss": "huber",
    "num_experts": 4,
    "noisy_gating": True,
    "k": 1,
    "CI": True,
}


class DUETConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)
