import torch
from src.nn.utils import softmax


class SimpleDecoder:
    def __init__(self, temperature=1.0, top_p=0.0, greedy=False):
        self.temperature = temperature
        self.top_p = top_p
        self.greedy = greedy

    def decode(self, logits: torch.Tensor):
        logits = logits[:, -1, :]
        probs = softmax(logits, dim=-1, temperature=self.temperature)

        if self.greedy:
            preds = probs.argmax(dim=-1, keepdim=True)  # (B, 1)
        else:
            # top p
            # Sort probs descending
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)

            remove = cumprobs > self.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False

            probs = sorted_probs.masked_fill(remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            dist = torch.distributions.Categorical(probs=probs)
            choice = dist.sample()
            B = torch.arange(probs.size(0), device=probs.device)
            preds = sorted_idx[B, choice].unsqueeze(-1)  # (B, 1)

        return preds
