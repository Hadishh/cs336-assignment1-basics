import torch
from src.decoder import SimpleDecoder
from src.tokenizer.bpe.bpe_tokenizer import Tokenizer


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

    def generate(self, init_tokens):
        pass
