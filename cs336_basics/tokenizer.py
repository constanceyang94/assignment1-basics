import regex as re
import os
# from .pretokenization_example import find_chunk_boundaries

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
    pre_tokenizer_counts = {} # Dictionary mapping pre-token (bytes) to count (int)
    merges = [] # List of merges, each merge is a tuple of bytes (<token1>, <token2>)
    vocab = {i: bytes([i]) for i in range(256)} # Initialize vocab with single byte tokens
    pair_counts = {} # Dictionary mapping pairs of tokens (tuple of bytes) to count (int)
    special_pattern = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    with open(input_path, "rb") as f:
        chunk = f.read().decode("utf-8", errors="ignore")
        for match in re.finditer(special_pattern, chunk):
            token = match.group()
            pre_tokenizer_counts[token] = pre_tokenizer_counts.get(token, 0) + 1
    for token in pre_tokenizer_counts.keys():
        for pair in zip(token, token[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    best_pair = max(pair_counts, key=pair_counts.get)
    merges.append(best_pair)
    vocab[len(vocab)] = bytes(best_pair)
    for _ in range(vocab_size - len(special_tokens) - 256):
        pair_counts = {}
        for token in pre_tokenizer_counts.keys():
            new_token = []
            for i in range(len(token) - 1):
                pair = (token[i], token[i+1])
                if pair == best_pair:
                    new_token.append(bytes(pair))
                    i += 2
                else:
                    new_token.append(token[i])
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)
    return vocab, merges

    
