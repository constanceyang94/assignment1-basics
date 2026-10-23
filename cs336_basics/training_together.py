import argparse
import numpy as np

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
    
    # Paramaters for data loading
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--training-path", type=str, help="input path to training data")
    parser.add_argument("--validation-path", type=str, help="input path to validation data")
    parser.add_argument("--checkpoint-path", type=str, help="path to store training checkpoint")
    
    args = parser.parse_args()
    return args

# Transformer LM and forward for each batch, need adam optimizer
def model_architecture(args: argparse.Namespace):
    transformer_lm = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, 
                                    args.num_heads, args.d_ff, args.rope_theta) 
    adam = AdamW(transformer_lm.parameters(), args.lr, args.weight_decay, [args.beta1, args.beta2], args.eps) 
    return transformer_lm, adam
    
def model_training(input: Int[Tensor, " batch_size sequence_length"], 
                   label: Int[Tensor, " batch_size"], 
                   transformer_lm: TransformerLM, 
                   adam: AdamW):
    adam.zero_grad()
    prediction = transformer_lm.forward(input)
    loss = cross_entropy(prediction, label)
    loss.backward()
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
    dataset = np.memmap(file_path, dtype=np.int8, mode='r')
    return dataset

def training_loop():
    args = parse_argument()
    transformer_lm, adam = model_architecture(args)
    training_data = load_dataset(args.training_path)
    validation_data = load_dataset(args.validation_path)
    for iteration in range(1000):
        input_batch, label_batch = get_batch(training_data, args.batch_size, args.context_length, 'cpu')
        training_loss = model_training(input_batch, label_batch, transformer_lm, adam)
        print(f"At iteration {iteration}, training loss is {training_loss.item()}")
        if iteration % 10 == 0:
            validation_batch, label_batch = get_batch(validation_data, args.batch_size, args.context_length, 'cpu')
            validation_loss = val_loss(validation_batch, label_batch, transformer_lm)
            print(f"At iteration {iteration}, validation loss is {validation_loss.item()}")
            save_checkpoint(transformer_lm, adam, iteration, args.checkpoint_path)
    return
    

if __name__ == "__main__":
    batch = 100
    lr = 0.1
    training_loop()