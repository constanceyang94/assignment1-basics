import argparse

from cs336_basics.modules import * 


# command line parsing
def parse_argument():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-3, help="Min Learning rate for lr schedule")
    parser.add_argument("--warmup-iters", type=int, default=10, help="Warm up iterations for lr schedule")
    parser.add_argument("--max-iters", type=int, default=100, help="Max iterations for lr schedule")
    
    # Hyper parameters for Adam
    parser.add_argument("--beta1", type=float, default=1e-3, help="beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=1e-3, help="beta2 for Adam optimizer")
    parser.add_argument("--eps", type=float, default=1e-3, help="eps for Adam optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="weight_decay for Adam optimizer")
    
    # Hyper parameter for TransformerLM
    parser.add_argument("--vocab-size", type=int, default=10000, help="vocab_size for TransformerLM")
    parser.add_argument("--context-length", type=int, default=256, help="context_length for TransformerLM")
    parser.add_argument("--d-model", type=int, default=512, help="d_model for TransformerLM")
    parser.add_argument("--d-ff", type=int, default=1344, help="d_ff for TransformerLM")
    parser.add_argument("--num_heads", type=int, default=16, help="num_heads for TransformerLM")
    parser.add_argument("--num-layers", type=int, default=4, help="num_layers for TransformerLM")
    parser.add_argument("--rope-theta", type=int, default=10000, help="rope_theta for TransformerLM")
    
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    return args

# Transformer LM and forward for each batch, need adam optimizer
def model_architecture(args):
    transformer_lm = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, 
                                    args.num_heads, args.d_ff, args.rope_theta, weights)
    
    return
    
# load training and validation dataset, and in batch way?

# Save checkpoint?

# Print training/validation loss

def main():
    args = parse_argument()
    

if __name__ == "__main__":
    batch = 100
    lr = 0.1
    print("abc")