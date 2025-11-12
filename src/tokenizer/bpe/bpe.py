from src.tokenizer.bpe.pre_tokenization import PreTokenizer


class BPETokenizer:
    def __update_stats(self, key, freq, occurance):
        self.stats[key] = self.stats.get(key, 0) + freq
        if key in self.indices:
            self.indices[key].append(occurance)
        else:
            self.indices[key] = [occurance]

    def __remove_stats(self, key, freq, occurance):
        self.stats[key] = self.stats.get(key) - freq
        if self.stats[key] == 0:
            self.stats.pop(key)

        self.indices[key].remove(occurance)

    def __init__(self, input_path, max_vocab_size, special_tokens):
        self.input_path = input_path
        self.max_vocab_size = max_vocab_size
        self.special_tokens = special_tokens

        self.indices = {}
        self.special_token_ids = [
            (tok, 256 + i) for i, tok in enumerate(special_tokens)
        ]

        self.vocab = {i: bytes((i,)) for i in range(256)}
        for token in special_tokens:
            self.vocab[len(self.vocab)] = token.encode("utf-8")

        with open(self.input_path, "rb") as f:
            pre_tok = PreTokenizer(f, special_tokens[0], num_processes=4)
            self.id2wstats = pre_tok.pre_tokenize()

        ## initializating stats
        self.stats = dict()
        self.indices = dict()
        for id in self.id2wstats:
            _, tokens, freq = self.id2wstats[id]
            for i in range(0, len(tokens) - 1):
                symbols = (tokens[i], tokens[i + 1])
                self.__update_stats(symbols, freq, id)

    def __create_new_tokens(self, tokens, max_symbols, tok_id):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == max_symbols[0]
                and tokens[i + 1] == max_symbols[1]
            ):
                new_tokens.append(tok_id)
                i += 2
                continue
            new_tokens.append(tokens[i])
            i += 1
        return new_tokens

    def __get_must_removed_indices(self, tokens, max_symbols):
        must_removed = set()
        # remove previous stats
        for i in range(2, len(tokens)):
            if tokens[i - 2] == max_symbols[0] and tokens[i - 1] == max_symbols[1]:
                must_removed.add(i - 1)
            if tokens[i] == max_symbols[1] and tokens[i - 1] == max_symbols[0]:
                must_removed.add(i - 2)

        return must_removed

    def __get_must_add_indices(self, new_tokens, tok_id):
        must_add = set()
        # update stats
        for i in range(len(new_tokens)):
            if new_tokens[i] == tok_id:
                if i > 0:
                    must_add.add(i - 1)
                if i + 1 < len(new_tokens):
                    must_add.add(i)
        return must_add

    def train_bpe(self):
        # we have initialized stats ready
        merges = []
        while len(self.vocab) < self.max_vocab_size:
            max_freq = max(self.stats.values())
            max_keys = [k for k, v in self.stats.items() if v == max_freq]
            max_keys = [
                (key, (self.vocab[key[0]], self.vocab[key[1]])) for key in max_keys
            ]
            max_keys = sorted(max_keys, key=lambda pair: pair[1], reverse=True)

            max_symbols = max_keys[0][0]
            merged_bytes = (self.vocab[max_symbols[0]], self.vocab[max_symbols[1]])
            merges.append(merged_bytes)
            tok_id = len(self.vocab)
            new_token = b"".join(merged_bytes)
            self.vocab[tok_id] = new_token
            word_ids = self.indices[max_symbols]

            for word_id in set(word_ids):
                word, tokens, freq = self.id2wstats[word_id]
                new_tokens = self.__create_new_tokens(tokens, max_symbols, tok_id)

                must_removed = self.__get_must_removed_indices(tokens, max_symbols)
                for i in must_removed:
                    key = (tokens[i], tokens[i + 1])
                    self.__remove_stats(key, freq, word_id)

                must_add = self.__get_must_add_indices(new_tokens, tok_id)
                for i in must_add:
                    key = (new_tokens[i], new_tokens[i + 1])
                    self.__update_stats(key, freq, word_id)

                self.id2wstats[word_id] = (word, tuple(new_tokens), freq)
            self.stats.pop(max_symbols)
            self.indices.pop(max_symbols)

        return merges


if __name__ == "__main__":
    from tests.test_train_bpe import test_train_bpe, test_train_bpe_special_tokens

    test_train_bpe_special_tokens(None)
    test_train_bpe()
