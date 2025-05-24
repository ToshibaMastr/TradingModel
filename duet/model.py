import torch
import torch.nn as nn
from einops import rearrange

from .layers.attention import AttentionLayer, FullAttention
from .layers.cluster import MixtureOfExperts
from .layers.encoder import Encoder, EncoderLayer
from .layers.masking import MahalanobisMask


class DUETModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cluster = MixtureOfExperts(config)
        self.num_variables = config.enc_in
        self.mask_generator = MahalanobisMask(config.seq_len)
        self.channels = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True, config.factor, attention_dropout=config.dropout
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model),
        )

        self.linear_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len), nn.Dropout(config.fc_dropout)
        )

    def extract_features(self, input):
        x = rearrange(input, "b l n -> (b n) l 1")
        features, importance = self.cluster(x)
        features = rearrange(features, "(b n) l 1 -> b l n", b=input.shape[0])
        return features, importance

    def forward(self, input):
        B, L, N = input.shape
        features, importance = self.extract_features(input)
        features = features.permute(0, 2, 1)

        if N > 1:
            input = input.permute(0, 2, 1)
            mask = self.mask_generator(input)
            features, attention = self.channels(features, mask)

        output = self.linear_head(features)
        output = output.permute(0, 2, 1)
        output = self.cluster.revin(output, "denorm")
        return output, importance
