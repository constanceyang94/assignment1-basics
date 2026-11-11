import argparse
import numpy as np
import time
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import find_chunk_boundaries

VOCAB_PATH = "/Users/zouxu/Desktop/CS336/Assignment1/bpe_output/tiny_story_vocab_10k.json"
MERGES_PATH = "/Users/zouxu/Desktop/CS336/Assignment1/bpe_output/tiny_story_merges_10k.txt"
TRAINING_PATH = "/Users/zouxu/Desktop/CS336/Assignment1/data/TinyStoriesV2-GPT4-train.txt"
VALIDATION_PATH = "/Users/zouxu/Desktop/CS336/Assignment1/data/TinyStoriesV2-GPT4-valid.txt"
OUTPUT_PATH = "/Users/zouxu/Desktop/CS336/Assignment1/train_eval_data"
SPECIAL_TOKENS = ["<|endoftext|>"]
CHUNK_SIZE = 20 * 1024 * 1024

def parse_argument():
    parser = argparse.ArgumentParser()
    
    # Parameters for tokenizers
    parser.add_argument("--vocab-path", type=str, default=VOCAB_PATH, 
                        help="path to tokenizer vocab")
    parser.add_argument("--merges-path", type=str, default=MERGES_PATH, 
                        help="path to tokenizer merges")
    parser.add_argument("--training-path", type=str, default=TRAINING_PATH, help="input path to training data")
    parser.add_argument("--validation-path", type=str, default=VALIDATION_PATH, help="input path to validation data")
    parser.add_argument("--output-path", type=str, default=OUTPUT_PATH, help="input path to validation data")
    
    args = parser.parse_args()
    return args

def get_tokenizer(args: argparse.Namespace):
    return Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, SPECIAL_TOKENS)

def preprocess(tokenizer: Tokenizer, input_path: str, output_path: str):
    with open(input_path, "rb") as fin, \
         open(output_path, "wb") as fout:
        while True:
            chunk = fin.read(CHUNK_SIZE)
            if not chunk:
                break

            if len(chunk) == CHUNK_SIZE:
                remainder = fin.readline()
                chunk += remainder

            token_ids = tokenizer.encode(chunk.decode("utf-8"))
            np.asarray(token_ids, dtype=np.uint32).tofile(fout)
    return

if __name__ == "__main__":
    args = parse_argument()
    tokenizer = get_tokenizer(args)
    training_output_path = Path(args.output_path) / f"{Path(args.training_path).stem}.bin"
    print("training output path is " + str(training_output_path))
    start = time.perf_counter()
    preprocess(tokenizer, args.training_path, training_output_path)
    elapsed = time.perf_counter() - start
    print(f"Encoding time for training data: {elapsed:.2f}s")
    
    validation_file_name = Path(args.output_path) / f"{Path(args.validation_path).stem}.bin"
    preprocess(tokenizer, args.validation_path, validation_file_name)
    