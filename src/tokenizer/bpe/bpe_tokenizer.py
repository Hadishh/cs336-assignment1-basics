import pickle as pkl
import regex as re


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.tok2id = {v: k for k, v in vocab.items()}
        self.merges = {
            (self.tok2id[merge[0]], self.tok2id[merge[1]]): idx
            for idx, merge in enumerate(merges)
        }
        if special_tokens:
            self.special_tokens = [tok.encode("utf-8") for tok in special_tokens]
        else:
            self.special_tokens = []
        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        for special_token in self.special_tokens:
            if not (special_token in self.tok2id):
                self.vocab[len(vocab)] = special_token
                self.tok2id[special_token] = len(self.tok2id)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, "rb") as f:
            vocab = pkl.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pkl.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _special_tokens(self, doc, special_tokens):
        if not special_tokens:
            return [doc]
        special_tokens = sorted(set(special_tokens), key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        parts = re.split(pattern, doc)

        parts = [p for p in parts if p]

        return parts

    def _pre_tokenize(self, chunk):
        pre_tokens = []
        parts = self._special_tokens(
            chunk, [tok.decode("utf-8") for tok in self.special_tokens]
        )
        for part in parts:
            if part.encode("utf-8") in self.special_tokens:
                pre_tokens.append(part.encode("utf-8"))
                continue
            for word in self.pattern.finditer(part):
                word = word.group().encode("utf-8")
                pre_tokens.append(word)

        return pre_tokens

    def encode(self, text):
        pre_tokens = self._pre_tokenize(text)
        tokens = []
        for word in pre_tokens:
            if word in self.special_tokens:
                tokens.append(self.tok2id[word])
                continue

            sequence = tuple([self.tok2id[word[i : i + 1]] for i in range(len(word))])

            while len(sequence) > 1:
                candidates = []
                for i in range(0, len(sequence) - 1):
                    new_key = (sequence[i], sequence[i + 1])
                    if new_key in self.merges:
                        candidates.append((new_key, i, self.merges[new_key]))

                if not candidates:
                    break

                # choose the earliest merge
                merge, idx, _ = sorted(candidates, key=lambda triple: triple[2])[0]

                merged_token = b"".join((self.vocab[merge[0]], self.vocab[merge[1]]))
                new_sequence = sequence[:idx] + (self.tok2id[merged_token],)
                if idx < len(sequence) - 2:
                    new_sequence += sequence[idx + 2 :]

                sequence = new_sequence

            tokens.extend(list(sequence))

        return tokens

    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids):
        ids = [self.vocab.get(id, -1) for id in ids]
        string = b"".join(ids).decode("utf-8", errors="replace")
        return string


if __name__ == "__main__":
    from tests.test_tokenizer import (
        test_tinystories_matches_tiktoken,
        test_overlapping_special_tokens,
        test_address_matches_tiktoken,
    )

    test_address_matches_tiktoken()
    test_overlapping_special_tokens()
    test_tinystories_matches_tiktoken()
