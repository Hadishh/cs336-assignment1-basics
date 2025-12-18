import torch
from src.decoder import SimpleDecoder
from src.tokenizer.bpe.bpe_tokenizer import Tokenizer
from src.nn.transformer import Transformer
from src.args import add_generation_args
import argparse
import numpy as np


def load_model(args, device):
    model = Transformer(
        args.vocab_size,
        args.context_length,
        args.nlayers,
        args.d_model,
        args.nheads,
        args.d_ff,
        args.rope_theta,
        device,
    )
    dict_ = torch.load(args.model_ckpt)
    model.load_state_dict(dict_["model"])

    return model


class SimpleGenerator:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        max_new_tokens,
        temperature=1.0,
        top_p=0.0,
        stop_token=b"<|endoftext|>",
        greedy=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        self.decoder = SimpleDecoder(
            temperature=temperature, top_p=top_p, greedy=greedy
        )
        self.stop_token = stop_token
        self.stop_token_id = tokenizer.tok2id[stop_token]

    def generate_single(self, init_tokens: torch.Tensor):
        current_state = init_tokens
        new_tokens = 0

        with torch.no_grad():
            while new_tokens < self.max_new_tokens:
                logits = self.model(current_state)  # (1, L)
                preds = self.decoder.decode(logits)  # (1, 1)
                current_state = torch.concat([current_state, preds], dim=1)
                if preds[0, 0] == self.stop_token_id:
                    break
                new_tokens += 1

        return self.tokenizer.decode(current_state.tolist()[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=6556)
    add_generation_args(parser)
    parser.add_argument("--input_text", type=str, default=None)

    args = parser.parse_args()
    if args.use_cuda:
        device = "cuda:0"
    else:
        device = "cpu"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)

    if args.input_text is not None:
        init_tokens = tokenizer.encode(args.input_text)
        init_tokens = torch.tensor(
            init_tokens, device=device, dtype=torch.int
        ).unsqueeze(0)
    else:
        exit(0)

    model = load_model(args, device)

    generator = SimpleGenerator(
        model,
        tokenizer,
        args.max_new_tokens,
        args.temperature,
        args.sampling_topp,
        greedy=(False if args.sampling else True),
    )

    print(generator.generate_single(init_tokens))
