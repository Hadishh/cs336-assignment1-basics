import regex as re
import os
from tqdm import tqdm


def split_paragraphs_special_tokens(doc, special_tokens):
    if not special_tokens:
        return [doc]
    special_tokens = sorted(set(special_tokens), key=len, reverse=True)
    pattern = "(?:" + "|".join(re.escape(tok) for tok in special_tokens) + ")"
    parts = re.split(pattern, doc)

    parts = [p for p in parts]

    return parts


def find_chunk_boundaries(
    file_path: str,
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
    file = open(file_path, "rb")
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
    file.close()
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(file_path, special_tokens, start, end, processor_id):
    """
    Pre-tokenization for parallel applications
    """
    fd = open(file_path, "rb")
    fd.seek(start)
    chunk = fd.read(end - start).decode("utf-8", errors="ignore")
    fd.close()

    # start_pretokenize
    pattern = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
    local_dict = {}
    parts = split_paragraphs_special_tokens(chunk, [special_tokens])
    for part in tqdm(parts, desc=f"Pretokenzing Proc {processor_id}."):
        for word in pattern.finditer(part):
            key = word.group().encode("utf-8")
            local_dict[key] = local_dict.get(key, 0) + 1

    return local_dict
