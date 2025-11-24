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
