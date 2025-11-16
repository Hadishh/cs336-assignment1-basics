from concurrent.futures import ProcessPoolExecutor

import src.tokenizer.bpe.utils as utils


class PreTokenizer:

    def __init__(
        self,
        file_path: str,
        split_special_token: str,
        num_processes: int = 4,
    ):
        self.file_path = file_path
        self.num_processes = num_processes
        self.split_special_token = split_special_token.encode("utf-8")

    def pre_tokenize(self) -> dict[bytes, int]:
        boundaries = utils.find_chunk_boundaries(
            self.file_path, self.num_processes, self.split_special_token
        )
        pid = 0
        arg_list = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            arg_list.append(
                (
                    self.file_path,
                    [self.split_special_token.decode("utf-8")],
                    start,
                    end,
                    pid,
                )
            )
            pid += 1
            # self.__pre_tokenize_chunk(start, end)

        assert len(arg_list) <= self.num_processes

        file_paths, special_tokens_list, starts, ends, pids = zip(*arg_list)
        with ProcessPoolExecutor(max_workers=16) as executor:
            final_results = list(
                executor.map(
                    utils.pre_tokenize_chunk,
                    file_paths,
                    special_tokens_list,
                    starts,
                    ends,
                    pids,
                )
            )

        global_vocab = dict()
        for local_dict in final_results:
            if local_dict is None:
                continue
            for key in local_dict:
                global_vocab[key] = global_vocab.get(key, 0) + local_dict[key]

        id2wstats = dict()
        for i, key in enumerate(global_vocab):
            id2wstats[i] = (key, tuple(key), global_vocab[key])

        return id2wstats
