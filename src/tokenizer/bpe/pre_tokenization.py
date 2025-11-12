import os
from typing import BinaryIO
import regex as re
import threading


class PreTokenizer:

    def __split_paragraphs_special_tokens(self, doc, special_tokens):
        pattern = "(?:" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
        parts = re.split(pattern, doc)

        parts = [p for p in parts if p]

        return parts

    def __find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(
            split_special_token, bytes
        ), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def __init__(
        self,
        file: BinaryIO,
        split_special_token: str,
        num_processes: int = 4,
    ):
        self.file = file
        self.num_processes = num_processes
        self.final_results = [None] * self.num_processes
        self.split_special_token = split_special_token.encode("utf-8")
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __pre_tokenize_chunk(self, start, end, processor_id):

        # lock_file
        self.file.seek(start)
        chunk = self.file.read(end - start).decode("utf-8", errors="ignore")
        # unlock_file

        # start_pretokenize
        local_dict = {}
        parts = self.__split_paragraphs_special_tokens(
            chunk, [self.split_special_token.decode("utf-8")]
        )
        for part in parts:
            for word in re.finditer(self.PAT, part):
                key = word.group().encode("utf-8")
                local_dict[key] = local_dict.get(key, 0) + 1

        self.final_results[processor_id] = local_dict
        return local_dict

    def pre_tokenize(self) -> dict[bytes, int]:
        boundaries = self.__find_chunk_boundaries(
            self.file, self.num_processes, self.split_special_token
        )
        pid = 0
        threads = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            thread = threading.Thread(
                target=self.__pre_tokenize_chunk, args=(start, end, pid)
            )
            threads.append(thread)
            thread.start()
            pid += 1
            # self.__pre_tokenize_chunk(start, end)

        assert len(threads) <= self.num_processes
        for thread in threads:
            thread.join()

        global_vocab = dict()
        for local_dict in self.final_results:
            if local_dict is None:
                continue
            for key in local_dict:
                global_vocab[key] = global_vocab.get(key, 0) + local_dict[key]

        id2wstats = dict()
        for i, key in enumerate(global_vocab):
            id2wstats[i] = (key, tuple(key), global_vocab[key])

        return id2wstats
