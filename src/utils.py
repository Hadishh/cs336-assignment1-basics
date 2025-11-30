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
