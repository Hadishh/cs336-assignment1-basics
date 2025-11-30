import math
import torch


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
