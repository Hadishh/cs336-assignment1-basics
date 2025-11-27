import torch
import einops

from src.nn.utils import CausalMultiHeadSelfAttention, RMSNorm, SwiGLU


class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta=1000.0,
        use_rope=False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, use_rope, device=device, dtype=dtype
        )
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        y = self.ln1(x)
        y = x + self.attn(y, token_positions)

        x_ffn = self.ln2(y)

        y = self.ffn(x_ffn) + y

        return y
