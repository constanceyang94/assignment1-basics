import ast
import json
import regex as re

from collections.abc import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """
        Initializes the BPE tokenizer with the given vocabulary, merges, and special tokens.

        Args:
            vocab: A dictionary mapping tokens to their corresponding IDs.
            merges: A list of merge operations for the BPE algorithm.
            special_tokens: A list of special tokens to be included in the tokenizer.
        """
        self.vocab = vocab
        self.encode_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.merges_dict = {v: i for i, v in enumerate(self.merges)}
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary
        and list of merges (in the same format that your BPE training code output)
        and (optionally) a list of special tokens. This method should accept the following additional parameters:

        Args:
            vocab_filepath: str
            merges_filepath: str
            special_tokens: list[str] | None = None
        """
        merges = []
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab: dict = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab.items()}
        with open(merges_filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for l in lines:
            items = l.strip("()").split(",")
            merges.append((ast.literal_eval(items[0]), ast.literal_eval(items[1])))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        encoded_list = []
        if self.special_tokens:
            special_pattern = "|".join(
                re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)
            )
        last_end = 0
        if self.special_tokens:
            for match in re.finditer(special_pattern, text):
                paragraph = text[last_end : match.start()]
                last_end = match.end()
                for token_match in re.finditer(PAT, paragraph):
                    token = token_match.group().encode("utf-8")
                    token_tuple = tuple(bytes([b]) for b in token)
                    self.encode_helper(token_tuple, encoded_list)
                encoded_list.append(self.encode_vocab[bytes(match.group().encode("utf-8"))])
        for token_match in re.finditer(PAT, text[last_end:]):
            token = token_match.group().encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in token)
            self.encode_helper(token_tuple, encoded_list)
        return encoded_list
            
    def encode_helper(self, token_tuple: tuple[bytes], encoded_list: list[int]):
        """encoder helper to encode small chunk of text"""
        find_merge = True
        find_merge_order = float('inf')
        while find_merge:
            find_merge = False
            find_merge_order = float('inf')
            merge = ()
            for i in range(1, len(token_tuple)):
                cur_tuple = (token_tuple[i-1], token_tuple[i])
                if cur_tuple in self.merges_dict:
                    if self.merges_dict[cur_tuple] < find_merge_order:
                        find_merge_order = self.merges_dict[cur_tuple]
                        merge = cur_tuple
                        find_merge = True
            if find_merge:
                merged_list = []
                for i in range(1, len(token_tuple)):
                    if i == 1:
                        merged_list.append(token_tuple[0])
                    cur_tuple = (token_tuple[i-1], token_tuple[i])
                    if cur_tuple == merge:
                        merged_list.pop()
                        merged_list.append(token_tuple[i-1] + token_tuple[i])
                    else:
                        merged_list.append(token_tuple[i])
                token_tuple = tuple(merged_list)
        for item in token_tuple:
            encoded_list.append(self.encode_vocab[item])
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for line in iterable:
            encoded_list = self.encode(line)
            yield from encoded_list   

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
        decoded_text = b""
        for id in ids:
            decoded_text += self.vocab[id]
        return decoded_text.decode("utf-8", errors="replace")
