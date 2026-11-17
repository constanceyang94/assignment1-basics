import argparse
import modal
import numpy as np
import wandb
import time

from cs336_basics.modules import * 
from datetime import datetime
from pathlib import Path


image = (
    modal.Image.debian_slim(python_version="3.12")
    # 1. Automatically reads your local pyproject.toml & uv.lock and runs `uv sync`!
    .uv_sync()
    # 2. Copies your local CS336 package (e.g. cs336_basics) into the container
    .add_local_python_source("cs336_basics")
)

app = modal.App("my-b200-trainer")
# 1. Connect Volume
training_vol = modal.Volume.from_name("b200-training")


# command line parsing
def parse_argument():
    training_path = "/workspace/train_eval_data/TinyStoriesV2-GPT4-train.bin"
    validation_path = "/workspace/train_eval_data/TinyStoriesV2-GPT4-valid.bin"
    checkpoint_path = "/workspace/checkpoints"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Min Learning rate for lr schedule")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Warm up iterations for lr schedule")
    parser.add_argument("--cosine-cycle-iters", type=int, default=5000, help="Max iterations for lr schedule")
    
    # Hyper parameters for Adam
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for Adam optimizer")
    parser.add_argument("--eps", type=float, default=1e-8, help="eps for Adam optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="weight_decay for Adam optimizer")
    
    # Hyper parameter for TransformerLM
    parser.add_argument("--vocab-size", type=int, default=10000, help="vocab_size for TransformerLM")
    parser.add_argument("--context-length", type=int, default=256, help="context_length for TransformerLM")
    parser.add_argument("--d-model", type=int, default=512, help="d_model for TransformerLM")
    parser.add_argument("--d-ff", type=int, default=1344, help="d_ff for TransformerLM")
    parser.add_argument("--num_heads", type=int, default=16, help="num_heads for TransformerLM")
    parser.add_argument("--num-layers", type=int, default=4, help="num_layers for TransformerLM")
    parser.add_argument("--rope-theta", type=int, default=10000, help="rope_theta for TransformerLM")
    
    # Paramaters for data loading
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--training-path", type=str, default=training_path, help="input path to training data")
    parser.add_argument("--validation-path", type=str, default=validation_path, help="input path to validation data")
    parser.add_argument("--checkpoint-path", type=str, default=checkpoint_path, help="path to store training checkpoint")
    
    args = parser.parse_args()
    return args

# Transformer LM and forward for each batch, need adam optimizer
def model_architecture(args: argparse.Namespace, device):
    transformer_lm = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, 
                                    args.num_heads, args.d_ff, args.rope_theta, device=device)
    adam = AdamW(transformer_lm.parameters(), args.lr, args.weight_decay, [args.beta1, args.beta2], args.eps) 
    return transformer_lm, adam
    
def model_training(input: Int[Tensor, " batch_size sequence_length"], 
                   label: Int[Tensor, " batch_size"], 
                   transformer_lm: TransformerLM, 
                   adam: AdamW,
                   lr: float):
    adam.zero_grad()
    prediction = transformer_lm.forward(input)
    loss = cross_entropy(prediction, label)
    loss.backward()
    for group in adam.param_groups:
        group["alpha"] = lr
    adam.step()
    return loss

def val_loss(val_input: Int[Tensor, " batch_size sequence_length"], 
             val_label: Int[Tensor, " batch_size"], 
             transformer_lm: TransformerLM):
    val_prediction = transformer_lm.forward(val_input)
    val_loss = cross_entropy(val_prediction, val_label)
    return val_loss

# load training and validation dataset
def load_dataset(file_path: str):
    dataset = np.memmap(file_path, dtype=np.uint32, mode='r')
    return dataset
 
@app.function(
    gpu="B200",
    image=image,
    volumes={
        "/workspace": training_vol,          # Read train / val Data
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=60 * 30,
)
def training_loop():
    # Initialize an experiment
    args = parse_argument()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    
    wandb.init(
        project="cs336-transformer",
        config={
            "learning_rate": args.lr,
            "min_lr": args.min_lr,
            "warmup_iters": args.warmup_iters,
            "cosine_cycle_iters": args.cosine_cycle_iters,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "eps": args.eps,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "steps": args.steps,
        }
    )

    transformer_lm, adam = model_architecture(args, device)
    torch.compile(transformer_lm)
    
    training_data = load_dataset(args.training_path)
    validation_data = load_dataset(args.validation_path)
    cur_date = datetime.now().strftime("%Y-%m-%d")
    checkpoint_path_dir = Path(args.checkpoint_path) / f"tiny_story_{cur_date}"
    checkpoint_path_dir.mkdir(parents=True, exist_ok=True)
    
    for step in range(args.steps):
        input_batch, label_batch = get_batch(training_data, args.batch_size, args.context_length, device)
        # torch.set_printoptions(threshold=float("inf"))
        # print("training input batch is " + str(len(input_batch)))
        lr = lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        training_loss = model_training(input_batch, label_batch, transformer_lm, adam, lr)
        print(f"At step {step}, training loss is {training_loss.item()}")
        # Log metrics
        wandb.log({
            "train_loss": training_loss,
            "step": step,
        })
        if step % 10 == 0:
            validation_batch, label_batch = get_batch(validation_data, args.batch_size, args.context_length, device)
            validation_loss = val_loss(validation_batch, label_batch, transformer_lm)
            print(f"At step {step}, validation loss is {validation_loss.item()}")
            checkpoint_path_cur = Path(checkpoint_path_dir) / f"training_tiny_story_{step}_{int(time.time())}.pt"
            save_checkpoint(transformer_lm, adam, step, checkpoint_path_cur)
            wandb.log({
                "validation_loss": validation_loss,
                "step": step,
            })
    wandb.finish()
    return

@app.local_entrypoint()
def main():
    print("Triggering Modal function...")
    training_loop.remote()
    print("Modal function finished!")