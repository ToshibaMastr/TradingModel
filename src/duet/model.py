import torch
import torch.nn as nn
from einops import rearrange

from .layers.linear_extractor_cluster import LinearExtractorCluster
from .utils.masked_attention import (
    AttentionLayer,
    Encoder,
    EncoderLayer,
    FullAttention,
    MahalanobisMask,
)


class DUETModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cluster = LinearExtractorCluster(config)
        self.CI = config.CI
        self.num_variables = config.enc_in
        self.mask_generator = MahalanobisMask(config.seq_len)
        self.channel_transformer = Encoder(
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
        if self.CI:
            x = rearrange(input, "b l n -> (b n) l 1")
            features, importance = self.cluster(x)
            features = rearrange(features, "(b n) l 1 -> b l n", b=input.shape[0])
        else:
            features, importance = self.cluster(input)

        return features, importance

    def forward(self, input):
        B, L, N = input.shape
        features, importance = self.extract_features(input)
        features = features.permute(0, 2, 1)

        if N > 1:
            input = input.permute(0, 2, 1)
            mask = self.mask_generator(input)
            feature, attention = self.channel_transformer(features, mask)
            output = self.linear_head(feature)
        else:
            output = self.linear_head(features)

        output = output.permute(0, 2, 1)
        output = self.cluster.revin(output, "denorm")
        return output, importance
