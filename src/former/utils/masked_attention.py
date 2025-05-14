import math
from math import sqrt

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )

        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)

                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        queries = queries * scale * 0.001

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            large_negative = math.log(1e10) * scale
            attention_mask = (attn_mask - 1) * large_negative

            scores = scores * attn_mask + attention_mask

        A = self.dropout(torch.softmax(scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous(), A


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class MahalanobisMask(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        frequency_size = input_size // 2 + 1
        self.A = nn.Parameter(
            torch.randn(frequency_size, frequency_size), requires_grad=True
        )

    def calculate_prob_distance(self, x):
        xf = torch.abs(torch.fft.rfft(x, dim=-1))
        x2 = xf.unsqueeze(2)
        x1 = xf.unsqueeze(1)
        diff = x1 - x2

        temp = diff @ self.A.T
        dist = temp.pow(2).sum(dim=-1)

        exp_dist = 1.0 / (dist + 3e-8)

        eye = torch.eye(exp_dist.shape[-1], device=exp_dist.device)
        exp_dist *= 1 - eye

        exp_max = exp_dist.max(dim=-1, keepdim=True).values.detach()
        p = exp_dist / (exp_max + 1e-8)

        p = (p + eye) * 0.99

        return p

    def bernoulli_gumbel_rsample(self, distribution_matrix):
        b, c, d = distribution_matrix.shape

        flatten = rearrange(distribution_matrix, "b c d -> (b c d) 1")
        complement = 1 - flatten

        logits = torch.log(flatten + 1e-8) - torch.log(complement + 1e-8)
        logits_pair = torch.cat([logits, -logits], dim=-1)

        samples = F.gumbel_softmax(logits_pair, hard=True)

        binary_mask = samples[..., 0].reshape(b, c, d)

        return binary_mask

    def forward(self, x):
        x = self.calculate_prob_distance(x)
        x = self.bernoulli_gumbel_rsample(x)
        mask = x.unsqueeze(1)
        cnt = torch.sum(mask, dim=-1)
        return mask
