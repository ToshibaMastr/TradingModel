import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
