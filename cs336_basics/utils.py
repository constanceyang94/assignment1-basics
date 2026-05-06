import json

def save_vocab(vocab: dict[int, bytes], output_path: str):
    """
    Save bpe tokenizer vocab to output_path. 
    Convert vocab value to hex since bytes cannot be serialized in json format.
    """
    transformed_vocab = {k: v.hex() for k, v in vocab.items()}
    with open(output_path, 'w') as f:
        json.dump(transformed_vocab, f)
    
def save_merges(merges: list[tuple[bytes, bytes]], output_path: str):
    """Save bpe tokenizer merges to output_path"""
    with open(output_path, 'w') as f:
        for merge in merges:
            f.write(f"{merge}\n")