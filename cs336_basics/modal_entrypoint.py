import modal

# 1. Package your local module into the image so Modal can import cs336_basics
image = (
    modal.Image.debian_slim()
    .pip_install("regex", "tqdm")
    .add_local_python_source("cs336_basics")  # Loads your local cs336_basics package
)

app = modal.App("OWT Tokenizer")
volume = modal.Volume.from_name("tokenizer-outputs", create_if_missing=True)

# input_path = "/Users/zouxu/Desktop/CS336/Assignment1/data/owt_train.txt"

@app.function(image=image,
                volumes={"/bpe": volume},
                cpu=8.0,       # Recommended: BPE training is heavy on CPU
                memory=65536,  # Recommended: Allocate sufficient RAM for vocab/merges in memory
                timeout=3600   # 1 hour timeout for large training runs)
            )
def train_bpe():
    # Write input owt dataset into modal remote instance
    import os
    from cs336_basics.train_bpe import train_bpe_tokenizer
    from cs336_basics.utils import save_vocab, save_merges
    import time
    
    print("--- [MODAL CONTAINER RUNNING] ---")
    os.makedirs("/bpe/output", exist_ok=True)
    
    modal_input_path = "/bpe/input/owt_train.txt"
    
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    print(f"Starting BPE training on {modal_input_path}...")
    start_time = time.time()
    vocab, merges = train_bpe_tokenizer(modal_input_path, vocab_size, special_tokens)
    end_time = time.time()
    print("Training bpe tokenizer for tiny story dataset used ", end_time - start_time, "seconds")

    vocab_file_path = "/bpe/output/owt_vocab.json"
    save_vocab(vocab, vocab_file_path)
                    
    merges_file_path = "/bpe/output/owt_merges.txt"
    save_merges(merges, merges_file_path)
    
    volume.commit()
    print("Saved results to volume successfully!")

@app.local_entrypoint()
def main():
    print("Triggering Modal function...")
    train_bpe.spawn()
    print("Modal function finished!")