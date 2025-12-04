import math
import torch
from typing import Iterable


def learning_rate_cosine_schedule(t, lr_max, lr_min, warmpus, annealing_iters):
    if t < warmpus:
        return lr_max * (t / warmpus)

    if warmpus <= t <= annealing_iters:
        return lr_min + 0.5 * (
            1 + math.cos(math.pi * (t - warmpus) / (annealing_iters - warmpus))
        ) * (lr_max - lr_min)

    if t > annealing_iters:
        return lr_min

    raise ValueError("LR Schedule: Invalid input.")


def gradient_clipping(params: Iterable[torch.nn.Parameter], M: float, eps=1e-6):
    global_norm = 0
    for param in params:
        if param.grad is None:
            continue
        l2 = torch.linalg.norm(param.grad)
        global_norm += l2 * l2
    global_norm = math.sqrt(global_norm)
    if global_norm < M:
        return

    for param in params:
        if param.grad is None:
            continue
        param.grad.data *= M / (global_norm + eps)


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out
):
    obj = {}

    obj["model"] = model.state_dict()
    obj["optimizer"] = optimizer.state_dict()
    obj["iteration"] = iteration

    torch.save(obj, out)

    return out


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    obj = torch.load(src)

    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]


def compute_steps(total_tokens, batch_size, context_length):
    return math.ceil(total_tokens / (batch_size * context_length))
