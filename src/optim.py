import torch
import math
from typing import Callable, Optional


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay, eps=1e-8, lr=1e-5, betas=(0.9, 0.95)):
        if lr < 0:
            raise ValueError(f"Invalid Learning rate: {lr}")

        if (betas[0] < 0 or betas[0] > 1) or (betas[1] < 0 or betas[1] > 1):
            raise ValueError(f"Invalid Beta: {betas}")

        defaults = {"lr": lr, "lambda": weight_decay, "epsilon": eps, "beta": betas}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["beta"]
            decay = group["lambda"]
            epsilon = group["epsilon"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.grad.data)

                if "v" not in state:
                    state["v"] = torch.zeros_like(p.grad.data)

                t = state.get("t", 1)
                m = state["m"]
                v = state["v"]

                grad = p.grad.data
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad * grad

                alpha_t = (
                    lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
                )

                p.data -= alpha_t * m / (v.sqrt() + epsilon)
                p.data -= lr * decay * p.data

                state["m"] = m
                state["t"] = t + 1
                state["v"] = v


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1, weight_decay=0.01)

    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
