import torch
import torch.nn as nn

from .enc_dec import series_decomp


class Extractor(nn.Module):  # ???
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.channels = configs.enc_in
        self.enc_in = 1

        self.decomp = series_decomp(configs.moving_avg)

        self.seasonal_conv = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels * self.pred_len,
            kernel_size=self.seq_len,
            groups=self.channels,
            bias=True,
        )
        self.trend_conv = nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels * self.pred_len,
            kernel_size=self.seq_len,
            groups=self.channels,
            bias=True,
        )

        init_val = 1.0 / self.seq_len
        for conv in (self.seasonal_conv, self.trend_conv):
            nn.init.constant_(conv.weight, init_val)
            nn.init.constant_(conv.bias, 0.0)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        B, L, C = x_enc.shape
        if B == 0:
            return x_enc.new_empty((0, self.pred_len, self.enc_in))

        seasonal, trend = self.decomp(x_enc)

        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        seasonal = self.seasonal_conv(seasonal)
        trend = self.trend_conv(trend)

        seasonal = seasonal.view(B, self.channels, self.pred_len)
        trend = trend.view(B, self.channels, self.pred_len)

        encoded = (seasonal + trend).permute(0, 2, 1)

        return encoded[:, -self.pred_len :, : self.enc_in]
