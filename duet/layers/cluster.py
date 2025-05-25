import torch
import torch.nn as nn
from einops import rearrange
from torch.distributions.normal import Normal

from .decomp import series_decomp
from .revin import RevIN


class SharedExtractor(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.d_model
        self.enc_in = 1

        self.decomp = series_decomp(configs.moving_avg)

        self.seasonal = nn.Linear(self.seq_len, self.pred_len, bias=False)
        self.trend = nn.Linear(self.seq_len, self.pred_len, bias=False)

        nn.init.constant_(self.seasonal.weight, 1.0 / self.seq_len)
        nn.init.constant_(self.trend.weight, 1.0 / self.seq_len)

    def forward(self, x_enc):
        if x_enc.shape[0] == 0:
            return x_enc.new_empty((0, self.pred_len, self.enc_in))

        seasonal, trend = self.decomp(x_enc)

        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        seasonal = self.seasonal(seasonal)
        trend = self.trend(trend)

        encoded = (seasonal + trend).permute(0, 2, 1)
        return encoded[:, -self.pred_len :, :]


class GatingNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config.seq_len
        num_experts = config.num_experts
        hidden_size = config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts, bias=False),
        )

    def forward(self, x):
        x = x.squeeze(-1)
        out = self.mlp(x)
        return out


class SparseDispatcher:
    def __init__(self, gates):
        self.gates = gates

        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self.expert_index = sorted_experts.split(1, dim=1)
        self.batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self.part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self.batch_index.flatten()]
        self.nonzero_gates = torch.gather(gates_exp, 1, self.expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self.batch_index]
        return torch.split(inp_exp, self.part_sizes, dim=0)

    def combine(self, expert_out, multiply=True):
        stitched = torch.cat(expert_out, 0)
        if multiply:
            stitched *= self.nonzero_gates.unsqueeze(-1)

        shape = list(expert_out[-1].shape)
        shape[0] = self.gates.size(0)
        zeros = torch.zeros(*shape, requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self.batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        return torch.split(self.nonzero_gates, self.part_sizes, dim=0)


class MixtureOfExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.noisy_gating = config.noisy_gating
        self.num_experts = config.num_experts
        self.input_size = config.seq_len
        self.k = config.k
        # instantiate experts
        self.experts = nn.ModuleList(
            [SharedExtractor(config) for _ in range(self.num_experts)]
        )
        self.weight = nn.Parameter(torch.eye(self.num_experts))
        self.gate = GatingNetwork(config)
        self.noise = GatingNetwork(config)

        self.n_vars = config.enc_in
        self.revin = RevIN(self.n_vars)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + 1e-10)

    def _prob_in_top_k(self, clean, noisy, noise_stddev, noisy_top):
        batch = clean.size(0)
        m = noisy_top.size(1)
        top_values_flat = noisy_top.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)

        return prob

    def noisy_top_k_gating(self, x, train, epsilon=1e-2):
        clean_logits = self.gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = self.noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + noise * noise_stddev
            logits = noisy_logits @ self.weight
        else:
            logits = clean_logits
        logits = self.softmax(logits)

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = top_k_logits / (
            top_k_logits.sum(1, keepdim=True) + 1e-6
        )  # normalization

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(
                clean_logits, noisy_logits, noise_stddev, top_logits
            )
        else:
            load = gates > 0
        return gates, load.sum(0)

    def forward(self, x, loss_coef=1):
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(gates)

        x_norm = rearrange(x, "(x y) l c -> x l (y c)", y=self.n_vars)
        x_norm = self.revin(x_norm, "norm")
        x_norm = rearrange(x_norm, "x l (y c) -> (x y) l c", y=self.n_vars)

        expert_inputs = dispatcher.dispatch(x_norm)

        gates = dispatcher.expert_to_gates()

        expert_outputs = [
            self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
        ]
        y = dispatcher.combine(expert_outputs)

        return y, loss
