import json
import time

from cs336_basics.train_bpe import train_bpe_tokenizer

input_path = "/Users/zouxu/Desktop/CS336/Assignment1/data/TinyStoriesV2-GPT4-train.txt"
vocab_size = 10000
special_tokens = ["<|endoftext|>"]

if __name__ == '__main__':
    start_time = time.time()
    vocab, merges = train_bpe_tokenizer(input_path, vocab_size, special_tokens)
    end_time = time.time()
    print("Training bpe tokenizer for tiny story dataset used ", end_time - start_time, "seconds")

    vocab_file_path = "/Users/zouxu/Desktop/CS336/Assignment1/bpe_output/tiny_story_vocab.txt"
    with open(vocab_file_path, 'w') as f:
        for key in vocab.keys():
            f.write(str(key) + ": " + str(vocab[key]) + "\n")
                    
    merges_file_path = "/Users/zouxu/Desktop/CS336/Assignment1/bpe_output/tiny_story_merges.txt"
    with open(merges_file_path, 'w') as f:
        for merge in merges:
            f.write(f"{merge}\n")