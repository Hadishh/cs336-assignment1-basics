import argparse
import os
import numpy as np
import wandb

from src.data.utils import data_loading
from src.nn.transformer import Transformer
from src.nn.utils import cross_entropy_loss
from src.optim import AdamW
from src.utils import (
    compute_steps,
    learning_rate_cosine_schedule,
    gradient_clipping,
    save_checkpoint,
    load_checkpoint,
)


def validation(config, model, valid_data):
    valid_iters = 0
    valid_loss = 0.0
    while valid_iters < config.valid_iters:
        x, y = data_loading(
            valid_data, config.batch_size, config.context_length, config.device
        )

        logits = model(x)
        loss = cross_entropy_loss(logits, y, reduction="mean")
        valid_loss += loss.item()
        valid_iters += 1

    return {"loss": valid_loss / valid_iters}


def train(config):
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

    while step <= total_steps:
        x, y = data_loading(
            train_data,
            batch_size=config.batch_size,
            context_length=config.context_length,
            device=config.device,
        )

        optimizer.zero_grad()

        logits = model(x)

        loss = cross_entropy_loss(logits, y, reduction="mean")
        loss.backward()

        # clipping
        gradient_clipping(model.parameters(), M=config.clip_norm)

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

        if valid_data is not None and config.valid_every % step == 0:
            valid_loss = validation(config, model, valid_data)
            wandb.log({"global_step": step, "valid/loss": valid_loss})
            print(f"Validation on {config.valid_iters} Iterations | Loss: {valid_loss}")
        if step % config.log_every == 0:
            wandb.log({"global_step": step, "train/loss": loss.item(), "train/lr": lr})
            print(f"Update: {step} | Loss: {loss.item()}  | lr : {lr}")

        if step % config.save_every == 0:
            save_path = os.path.join(config.output_dir, f"checkpoint_{step}.pt")
            save_checkpoint(model, optimizer, step, save_path)

        step += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--adamw_beta1", type=float, default=0.9)
    parser.add_argument("--adamw_beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--clip_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=1500)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--lr_schedule", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--annealing_iters", type=int, defulat=20000)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--total_tokens_processed", type=int, default=327_680_000)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--wadb_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--valid_data", type=str, default=None)
    parser.add_argument("--valid_iters", type=int, defalut=100)
    parser.add_argument("--valid_every", type=int, default=1000)

    args = parser.parse_args()

    if args.use_cuda:
        device = "cuda:0"
    else:
        device = "cpu"
    args = vars(args)
    args["device"] = device

    wandb.init(project=args.wandb_name, config=args)
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    if args["valid_data"]:
        wandb.define_metric("valid/*")

    config = wandb.config

    train(config)
