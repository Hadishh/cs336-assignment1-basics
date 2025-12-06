import math
import torch
from typing import Iterable
import yaml


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


def load_config(args, arg_types):
    """Merge YAML config into argparse args.

    Priority: YAML value (if present) > CLI / default.
    Types are coerced to match argparse's types.
    """
    cfg = vars(args).copy()

    if args.config is None:
        return cfg

    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    for key, yaml_val in yaml_cfg.items():
        if key not in cfg:
            continue  # ignore unknown keys

        expected_type = arg_types.get(key)

        if expected_type is not None:
            try:
                # already right type?
                if isinstance(yaml_val, expected_type):
                    cfg[key] = yaml_val
                else:
                    cfg[key] = expected_type(yaml_val)
            except Exception:
                # fallback: leave it as-is
                cfg[key] = yaml_val
        else:
            cfg[key] = yaml_val

    return cfg
