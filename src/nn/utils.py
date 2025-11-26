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
        self.weight = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor):
        return einops.einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        e = torch.zeros(
            size=(num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        e = torch.nn.init.trunc_normal_(e, mean=0, std=1, a=-3, b=3)

        self.weight = torch.nn.Parameter(e)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        g = torch.ones(size=(d_model,), device=device, dtype=dtype)

        self.weight = torch.nn.Parameter(g)
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
            x, self.weight, "batch seq_len d_model, d_model -> batch seq_len d_model"
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

        cos = einops.einsum(x, cos, "... seq_len d, ... seq_len d -> ... seq_len d")
        sin = einops.einsum(temp, sin, "... seq_len d, ... seq_len d -> ... seq_len d")

        return sin + cos


def softmax(in_features: torch.Tensor, dim: int):

    max_ = in_features.max(dim=dim, keepdim=True).values
    v = in_features - max_

    exp_ = torch.exp(v)
    sum_ = torch.sum(exp_, dim=dim, keepdim=True)

    return exp_ / sum_


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None
):

    d_k = K.shape[-1]
    attn = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")

    attn = attn / math.sqrt(d_k)

    if mask is not None:
        inf = torch.where(mask, torch.tensor(0), -torch.inf).to(
            device=Q.device, dtype=Q.dtype
        )
        attn = attn + inf

    attn = softmax(attn, dim=-1)

    attn = einops.einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")

    return attn


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        max_sequence_len,
        theta=1000.0,
        use_rope=False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.use_rope = use_rope

        d_k = d_v = d_model // num_heads

        self.k_proj = Linear(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.q_proj = Linear(num_heads * d_k, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(num_heads * d_v, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, num_heads * d_v, device=device, dtype=dtype)

        self.RoPE = RotaryPositionalEmbeddings(
            d_k, theta=theta, max_seq_len=max_sequence_len, device=device
        )

    def __multihead_attention(self, Q, K, V, token_positions=None):
        # Q shape = (b, seq_len, d_model)
        d_k = d_v = self.d_model // self.num_heads
        seq_len = Q.shape[-2]
        batch_s = Q.shape[0]
        Q = einops.rearrange(
            Q, "... seq (h dk) -> ... h seq dk", h=self.num_heads, dk=d_k
        )
        K = einops.rearrange(
            K, "... seq (h dk) -> ... h seq dk", h=self.num_heads, dk=d_k
        )
        V = einops.rearrange(
            V, "... seq (h dv) -> ... h seq dv", h=self.num_heads, dv=d_v
        )

        if self.use_rope:
            Q = einops.rearrange(Q, "b h seq dk -> (b h) seq dk")
            Q = self.RoPE(Q, token_positions)
            Q = einops.rearrange(
                Q, "(b h) seq dk -> b h seq dk", b=batch_s, h=self.num_heads
            )

            K = einops.rearrange(K, "b h seq dk -> (b h) seq dk")
            K = self.RoPE(K, token_positions)
            K = einops.rearrange(
                K, "(b h) seq dk -> b h seq dk", b=batch_s, h=self.num_heads
            )

        mask = torch.ones((seq_len, seq_len), device=Q.device)
        mask = 1 - torch.triu(mask, diagonal=1)
        multiheadattn = scaled_dot_product_attention(Q, K, V, mask.bool())

        out = einops.rearrange(multiheadattn, "... h seq d_v -> ... seq (h d_v)")

        return out

    def forward(self, x, token_positions=None):

        Q = self.q_proj(x)  # (... seq_len d_model)
        K = self.k_proj(x)  # (... seq_len d_model)
        V = self.v_proj(x)  # (... seq_len d_model)

        multihead = self.__multihead_attention(Q, K, V, token_positions)

        out = self.output_proj(multihead)

        return out
