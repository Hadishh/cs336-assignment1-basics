import torch
from src.nn.tranfromer_layer import TransformerLayer
from src.nn.utils import RotaryPositionalEmbeddings, Embedding, RMSNorm, Linear


class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        theta=1000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.rope = RotaryPositionalEmbeddings(
            d_k=d_model // num_heads,
            theta=theta,
            max_seq_len=context_length,
            device=device,
        )

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype
        )

        self.layers = [
            TransformerLayer(d_model, num_heads, d_ff, self.rope, device, dtype)
            for _ in range(num_layers)
        ]

        self.layers = torch.nn.ModuleList(self.layers)

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        self.lm_head = Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype
        )

    def forward(self, x, token_positions):

        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)

        x = self.lm_head(x)

        return x
