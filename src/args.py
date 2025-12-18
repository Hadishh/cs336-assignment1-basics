def add_tokenizer_args(parser):
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--merges", required=True)
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"])
    return parser


def add_model_args(parser):
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--vocab_size", type=int, required=True)
    return parser


def add_training_args(parser):
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=32)
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
    parser.add_argument("--annealing_iters", type=int, default=20000)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--total_tokens_processed", type=int, default=327_680_000)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--valid_data", type=str, default=None)
    parser.add_argument("--valid_iters", type=int, default=100)
    parser.add_argument("--valid_every", type=int, default=1000)
    parser.add_argument("--save_best_n", type=int, default=5)
    return parser


def add_generation_args(parser):
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--sampling_topp", type=float, default=-1.0)
    parser.add_argument("--sampling_topk", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    add_tokenizer_args(parser)
    add_model_args(parser)

    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, required=True)
    return parser
