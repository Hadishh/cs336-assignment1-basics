import torch
from src.nn.utils import softmax


class SimpleDecoder:
    def __init__(self, temperature=1.0, top_p=0.0, greedy=False):
        self.temperature = temperature
        self.top_p = top_p
        self.greedy = False

    def decode(self, logits: torch.Tensor):
        logits = logits[:, -1, :]
        probs = softmax(logits, dim=-1, temperature=self.temperature)

        # top p
        probs[probs < self.top_p] = 0

        probs = probs / probs.sum(dim=-1, keepdim=True)

        if self.greedy:
            preds = probs.argmax(dim=-1, keepdim=True)  # (B, 1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            preds = dist.sample()  # (B, 1)

        return preds
