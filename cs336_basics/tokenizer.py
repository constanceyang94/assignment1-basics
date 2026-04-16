import regex as re
import os

# from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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
        vocab[len(vocab)] = bytes(special_token.encode("utf-8"))
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
    pre_tokenizer_counts = {}
    special_pattern = "|".join(
        re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
    )
    last_end = 0
    with open(input_path, "rb") as f:
        chunk = f.read().decode("utf-8", errors="ignore")
        if len(special_tokens) >= 1:
            for match in re.finditer(special_pattern, chunk):
                paragraph = chunk[last_end : match.start()]
                last_end = match.end()
                pretokenize_helper(pre_tokenizer_counts, paragraph)
        paragraph = chunk[last_end : len(chunk)]
        pretokenize_helper(pre_tokenizer_counts, paragraph)
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
