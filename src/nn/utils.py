import torch
import einops
import math


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        sigma = math.sqrt(2 / (in_features + out_features))
        w = torch.zeros(size=(out_features, in_features), dtype=dtype, device=device)
        w = torch.nn.init.trunc_normal_(
            w, mean=0, std=sigma * sigma, a=-3 * sigma, b=3 * sigma
        )
        self.weights = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor):
        return einops.einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        e = torch.zeros(
            size=(num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        e = torch.nn.init.trunc_normal_(e, mean=0, std=1, a=-3, b=3)

        self.embeddings = torch.nn.Parameter(e)

    def forward(self, token_ids: torch.Tensor):
        return self.embeddings[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        g = torch.ones(size=(d_model,), device=device, dtype=dtype)

        self.g = torch.nn.Parameter(g)
        self.epsilon = torch.Tensor([eps])

    def forward(self, x: torch.Tensor):
        x_type = x.dtype

        x = x.to(dtype=torch.float32)

        rms = einops.reduce(
            x * x, "batch seq_len d_model -> batch seq_len 1", reduction="mean"
        )
        rms = rms + self.epsilon
        rms = torch.sqrt(rms)

        x = x / rms

        return einops.einsum(
            x, self.g, "batch seq_len d_model, d_model -> batch seq_len d_model"
        ).to(dtype=x_type)


class SiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sig = torch.nn.functional.sigmoid(x)

        return x * sig


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        self.L1 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.L2 = Linear(
            in_features=d_ff, out_features=d_model, device=device, dtype=dtype
        )
        self.L3 = Linear(
            in_features=d_model, out_features=d_ff, device=device, dtype=dtype
        )
        self.SiLU = SiLU()

    def forward(self, x: torch.Tensor):

        x1 = self.SiLU(self.L1(x))
        x3 = self.L3(x)

        x2 = einops.einsum(x1, x3, "... dff, ... dff -> ... dff")

        return self.L2(x2)


class RotaryPositionalEmbeddings(torch.nn.Module):
    def __init__(self, d_k, theta, max_seq_len, device=None):
        super().__init__()

        r = [[] for i in range(max_seq_len)]
        self.theta = theta
        seq_len_ind = torch.arange(0, max_seq_len, device=device).unsqueeze(1)
        k_theta = torch.arange(0, d_k // 2, device=device).unsqueeze(0)

        k_theta = self.theta ** (2 * k_theta / d_k)
        k_theta = k_theta.repeat(max_seq_len, 1)
        k_theta = seq_len_ind / k_theta
        k_theta = einops.repeat(k_theta, "L k -> L (k r)", r=2)  # theta matrix is ready

        sin = torch.sin(k_theta)
        cos = torch.cos(k_theta)

        # of shape (max_seq_len, d_k)
        self.register_buffer(name="cos", tensor=cos, persistent=False)
        self.register_buffer(name="sin", tensor=sin, persistent=False)

    def forward(self, x, token_inds):

        cos = self.cos[token_inds]  # (seq_len, d_k)
        sin = self.sin[token_inds]  # (seq_len, d_k)

        temp = einops.rearrange(
            x, "... (p a) -> ... p a", a=2
        )  # (..., seq_len, d_k//2, 2)
        temp = temp[..., [1, 0]]
        temp = einops.rearrange(temp, "... p a -> ... (p a)")  # (..., seq_len, d_k)
        mul = 1 - 2 * (
            (1 + torch.arange(temp.shape[-1], device=x.device)) % 2
        )  # [-1, 1, -1, 1, ..]
        temp = temp * mul

        cos = einops.einsum(x, cos, "... seq_len d, seq_len d -> ... seq_len d")
        sin = einops.einsum(temp, sin, "... seq_len d, seq_len d -> ... seq_len d")

        return sin + cos


def softmax(in_features: torch.Tensor, dim: int):

    max_ = in_features.max(dim=dim, keepdim=True).values
    v = in_features - max_

    exp_ = torch.exp(v)
    sum_ = torch.sum(exp_, dim=dim, keepdim=True)

    return exp_ / sum_
