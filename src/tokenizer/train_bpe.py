import argparse
from src.tokenizer.bpe.bpe import BPETokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--special_tokens", nargs="+", default=["<|endoftext|>"])

    args = parser.parse_args()

    tokenizer = BPETokenizer(
        args.input_file, args.vocab_size, args.special_tokens, args.num_proc
    )

    tokenizer.train_bpe()

    tokenizer.save_state(args.output_dir)
