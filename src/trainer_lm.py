import argparse
import os
import numpy as np
import wandb

import einops
from src.data.utils import data_loading, valid_data_loading
from src.nn.transformer import Transformer
from src.nn.utils import cross_entropy_loss
from src.optim import AdamW
from src.utils import (
    compute_steps,
    learning_rate_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
    load_config,
)
from src.args import add_model_args, add_training_args


def validation(config, model, valid_data):
    valid_loss = 0.0

    for _ in range(config.valid_iters):

        x, y = data_loading(
            valid_data, config.valid_batch_size, config.context_length, config.device
        )
        logits = model(x)
        logits_flat = einops.rearrange(logits, "b l v -> (b l) v")
        y_flat = einops.rearrange(y, "b l -> (b l)")
        loss = cross_entropy_loss(logits_flat, y_flat, reduction="mean")
        valid_loss += loss.item()

    return {"loss": valid_loss / config.valid_iters}


def save_best_n_valid_loss(
    config, best_valid_losses, valid_loss, model, optimizer, step
):
    if len(best_valid_losses) < config.save_best_n:
        save_path = os.path.join(config.output_dir, f"checkpoint_{valid_loss:.3f}.pt")
        save_checkpoint(model, optimizer, step, save_path)
        best_valid_losses.append((valid_loss, save_path))
        best_valid_losses.sort(key=lambda x: x[0])

    elif valid_loss < best_valid_losses[-1][0]:
        _, worst_path = best_valid_losses[-1]
        if os.path.exists(worst_path):
            os.remove(worst_path)

        save_path = os.path.join(config.output_dir, f"checkpoint_{valid_loss:.3f}.pt")
        save_checkpoint(model, optimizer, step, save_path)

        best_valid_losses[-1] = (valid_loss, save_path)
        best_valid_losses.sort(key=lambda x: x[0])

    return best_valid_losses


def train(config):
    best_valid_losses = []
    train_data = np.load(config.train_data, mmap_mode="r")
    if config.valid_data:
        valid_data = np.load(config.valid_data, mmap_mode="r")
    else:
        valid_data = None

    lr_scheduler = None
    if config.lr_schedule == "cosine":
        lr_scheduler = learning_rate_cosine_schedule
    model = Transformer(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.nlayers,
        num_heads=config.nheads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        theta=config.rope_theta,
        device=config.device,
    )

    optimizer = AdamW(
        params=model.parameters(),
        weight_decay=config.weight_decay,
        betas=(config.adamw_beta1, config.adamw_beta2),
        lr=config.learning_rate,
    )

    total_steps = compute_steps(
        config.total_tokens_processed, config.batch_size, config.context_length
    )

    if config.load_checkpoint is not None:
        step = load_checkpoint(config.load_checkpoint, model, optimizer)
    else:
        step = 1

    print("Initiated with total steps: ", total_steps)
    while step <= total_steps:
        x, y = data_loading(
            train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )

        optimizer.zero_grad()

        logits = model(x)

        logits_flat = einops.rearrange(logits, "b l v -> (b l) v")
        y_flat = einops.rearrange(y, "b l -> (b l)")

        loss = cross_entropy_loss(logits_flat, y_flat, reduction="mean")

        loss.backward()

        # clipping
        gnorm = gradient_clipping(model.parameters(), M=config.clip_norm)

        lr = config.learning_rate
        if lr_scheduler is not None:
            lr = learning_rate_cosine_schedule(
                step,
                config.lr_max,
                config.lr_min,
                config.warmup,
                config.annealing_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        optimizer.step()

        if valid_data is not None and step % config.valid_every == 0:
            valid_loss = validation(config, model, valid_data)
            valid_loss = valid_loss["loss"]
            wandb.log({"global_step": step, "valid/loss": valid_loss})
            print(f"Validation | Loss: {valid_loss}")
            best_valid_losses = save_best_n_valid_loss(
                config, best_valid_losses, valid_loss, model, optimizer, step
            )

        if step % config.log_every == 0:
            wandb.log(
                {
                    "global_step": step,
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "train/gnorm": gnorm,
                }
            )
            print(f"Update: {step} | Loss: {loss.item():.5f}  | lr : {lr:.2e}")

        if step % config.save_every == 0:
            save_path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
            save_checkpoint(model, optimizer, step, save_path)

        step += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    add_model_args(parser)
    add_training_args(parser)

    args = parser.parse_args()
    arg_types = {a.dest: a.type for a in parser._actions if a.type is not None}
    args = load_config(args, arg_types)
    if args["use_cuda"]:
        device = "cuda:0"
    else:
        device = "cpu"
    args["device"] = device

    wandb.init(project=args["wandb_project"], name=args["wandb_name"], config=args)
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    if args["valid_data"]:
        wandb.define_metric("valid/*", step_metric="global_step")

    config = wandb.config
    os.makedirs(config.output_dir, exist_ok=True)
    train(config)
