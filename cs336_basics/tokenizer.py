import regex as re
import os

from multiprocessing import Pool
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DELIMITER = "<|endoftext|>"


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens):
        """
        Initializes the BPE tokenizer with the given vocabulary, merges, and special tokens.

        Args:
            vocab: A dictionary mapping tokens to their corresponding IDs.
            merges: A list of merge operations for the BPE algorithm.
            special_tokens: A list of special tokens to be included in the tokenizer.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens


def find_chunk_boundaries(
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


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Plan:
    # 1. Pre-tokenizer
    # 2. Compute BPE merges
    # 3. Change pre-tokenizer to multiprocessing using pre-tokenization example
    pre_tokenizer_counts = {}  # Dictionary mapping pre-token (bytes) to count (int)
    merges = []  # List of merges, each merge is a tuple of bytes (<token1>, <token2>)
    vocab = {
        i: bytes([i]) for i in range(256)
    }  # Initialize vocab with single byte tokens
    if vocab_size - len(special_tokens) < 256:
        return vocab, merges
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")
    pre_tokenizer_counts = pretokenize(input_path, special_tokens)
    for _ in range(vocab_size - len(special_tokens) - 256):
        compute_single_bpe_merge(pre_tokenizer_counts, merges, vocab)
    return vocab, merges


def pretokenize(input_path: str | os.PathLike, special_tokens: list[str]):
    """
    Given the input path to BPE tokenizer training data and special tokens, splits around special
    tokens, and applies the GPT-2 regex pattern to count pre-token occurrences.

    Args:
        input_path: input path to training data.
        special_tokens: to split input data.
    """
    pre_tokenizer_counts_dicts = []
    special_pattern = "|".join(
        re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
    )

    with open(input_path, "rb") as f:
        num_processes = 8
        parameter_list = []
        boundaries = find_chunk_boundaries(f, num_processes, DELIMITER.encode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            parameter_list.append([input_path, special_pattern, start, end])

    with Pool(processes=num_processes) as pool:
        pre_tokenizer_counts_dicts = pool.starmap(multiprocess_helper, parameter_list)

    pre_tokenizer_counts = {}
    for count_dict in pre_tokenizer_counts_dicts:
        for key in count_dict:
            pre_tokenizer_counts[key] = (
                pre_tokenizer_counts.get(key, 0) + count_dict[key]
            )
        # with Pool(processes=4) as pool:
        #     pool.map(a, b)
    return pre_tokenizer_counts


def multiprocess_helper(
    input_path: str, special_pattern: str, start: int, end: int
) -> dict[tuple, int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        pre_tokenizer_counts = {}
        last_end = 0
        for match in re.finditer(special_pattern, chunk):
            paragraph = chunk[last_end : match.start()]
            last_end = match.end()
            pretokenize_helper(pre_tokenizer_counts, paragraph)
        pretokenize_helper(pre_tokenizer_counts, chunk[last_end:])
        return pre_tokenizer_counts


def pretokenize_helper(
    pre_tokenizer_counts: dict[tuple[bytes, ...], int], paragraph: str
):
    """
    pretokenize_helper which takes paragraph as input and mutate the pre_tokenizer_counts in place.
    """
    for token_match in re.finditer(PAT, paragraph):
        token = token_match.group().encode("utf-8")
        token_tuple = tuple(bytes([b]) for b in token)
        pre_tokenizer_counts[token_tuple] = pre_tokenizer_counts.get(token_tuple, 0) + 1


def compute_single_bpe_merge(
    pre_tokenizer_counts: dict[tuple[bytes, ...], int],
    merges: list[tuple[bytes, bytes]],
    vocab: dict[int, bytes],
):
    """
    Single BPE merge during tokenizer. Merge the token pair with the highest frequency and mutate
    pre_tokenizer_counts in place after the merge.
    The function modifies merge and vocab in place as well.
    """
    pair_counts = {}
    for token in pre_tokenizer_counts.keys():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + pre_tokenizer_counts[token]
    best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
    merges.append(best_pair)
    vocab[len(vocab)] = best_pair[0] + best_pair[1]
    for token in list(pre_tokenizer_counts.keys()):
        token_tuple_new = []
        i = 0
        while i < len(token):
            if i == len(token) - 1:
                token_tuple_new.append(token[i])
                break
            pair = (token[i], token[i + 1])
            if pair == best_pair:
                token_tuple_new.append(pair[0] + pair[1])
                i += 2
            else:
                token_tuple_new.append(token[i])
                i += 1
        token_tuple_new = tuple(token_tuple_new)
        if token_tuple_new != token:
            pre_tokenizer_counts[token_tuple_new] = pre_tokenizer_counts[token]
            del pre_tokenizer_counts[token]
