import pickle as pkl
import regex as re
from array import array


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
            self.decoded_special_tokens = special_tokens
            self.decoded_special_tokens = sorted(
                set(special_tokens), key=len, reverse=True
            )
            pattern = (
                "("
                + "|".join(re.escape(tok) for tok in self.decoded_special_tokens)
                + ")"
            )
            self.special_pattern = re.compile(pattern)
        else:
            self.special_tokens = []
            self.decoded_special_tokens = []
            self.special_pattern = None

        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        for special_token in self.special_tokens:
            if not (special_token in self.tok2id):
                self.vocab[len(vocab)] = special_token
                self.tok2id[special_token] = len(self.tok2id)

        self.merge_to_id = dict()
        for (left_id, right_id), idx in list(self.merges.items()):
            merged_token = self.vocab[left_id] + self.vocab[right_id]
            merged_id = self.tok2id[merged_token]
            self.merge_to_id[(left_id, right_id)] = merged_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):

        with open(vocab_filepath, "rb") as f:
            vocab = pkl.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pkl.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _special_tokens(self, doc, special_tokens):
        if not self.special_pattern:
            yield doc
            return
        last = 0
        for m in self.special_pattern.finditer(doc):
            if m.start() > last:
                yield doc[last : m.start()]
            yield m.group(0)
            last = m.end()

        if last < len(doc):
            yield doc[last:]

    def _pre_tokenize(self, chunk):
        for part in self._special_tokens(chunk, self.decoded_special_tokens):
            if part in self.decoded_special_tokens:
                yield part.encode("utf-8")
                continue
            for word in self.pattern.finditer(part):
                word = word.group().encode("utf-8")
                yield word

    def encode(self, text):
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable):
        for text in iterable:
            for word in self._pre_tokenize(text):
                if word in self.special_tokens:
                    yield self.tok2id[word]
                    continue

                sequence = [self.tok2id[word[i : i + 1]] for i in range(len(word))]

                while len(sequence) > 1:
                    best_candidate = (None, -1, float("inf"))
                    for i in range(0, len(sequence) - 1):
                        new_key = (sequence[i], sequence[i + 1])
                        if (
                            new_key in self.merges
                            and best_candidate[-1] > self.merges[new_key]
                        ):
                            best_candidate = (new_key, i, self.merges[new_key])

                    if not best_candidate[0]:
                        break

                    # choose the earliest merge
                    merge, idx, _ = best_candidate

                    merged_id = self.merge_to_id[merge]

                    sequence[idx : idx + 2] = [merged_id]

                yield from sequence

    def decode(self, ids):
        ids = [self.vocab.get(id, -1) for id in ids]
        string = b"".join(ids).decode("utf-8", errors="replace")
        return string


if __name__ == "__main__":
    from tests.test_tokenizer import (
        test_tinystories_matches_tiktoken,
        test_overlapping_special_tokens,
        test_address_matches_tiktoken,
        test_encode_memory_usage,
    )

    test_encode_memory_usage()
    test_address_matches_tiktoken()
    test_overlapping_special_tokens()
    test_tinystories_matches_tiktoken()
