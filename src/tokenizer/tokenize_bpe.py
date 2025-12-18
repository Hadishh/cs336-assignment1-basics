import argparse

from src.tokenizer.bpe.bpe_tokenizer import Tokenizer
import numpy as np
from tqdm import tqdm
import os
from src.args import add_tokenizer_args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--flatten", action="store_true")

    parser = add_tokenizer_args(parser)

    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_tokens)

    with open(args.input_file) as f:
        text = f.read()

    tokens = []
    for tok in tqdm(tokenizer.encode_iterable([text])):
        tokens.append(tok)

    tokens = np.array(tokens, dtype=np.uint16)

    if args.flatten:
        tokens = tokens.flatten()
    dirname = os.path.dirname(args.out)
    os.makedirs(dirname, exist_ok=True)
    with open(args.out, "wb") as f:
        np.save(f, tokens)

    total_bytes = len(bytes(text, encoding="utf-8"))
    compression_ratio = total_bytes / len(tokens)
    print(f"Compression Ratio: {compression_ratio:0.2f}")
