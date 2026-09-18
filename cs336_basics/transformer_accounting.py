def compute_transformer_flops(vocab_size: int, 
                              context_length: int,
                              num_layers: int,
                              d_model: int,
                              num_heads: int,
                              d_ff: int):
    attention_flops = 4 * (2 * context_length * d_model * d_model) + \
                        2 * (2 * context_length * context_length * d_model)
    fnn_flops = 3 * (2 * context_length * d_model * d_ff)
    total_flops_in_one_block = attention_flops + fnn_flops
    print("attention_flops is " + str(attention_flops))
    print("fnn_flops is " + str(fnn_flops))
    print("Ratio of attention_flops is " + str(attention_flops / total_flops_in_one_block))
    print("Ratio of fnn_flops is " + str(fnn_flops / total_flops_in_one_block))
    final_linear = 2 * vocab_size * d_model * context_length
    print("final_linear is " + str(final_linear))
    total_flops = num_layers * (attention_flops + fnn_flops) + final_linear
    print("total flops is " + str(total_flops))
    print("Ratio of final linear projection is " + str(final_linear / total_flops))
    return total_flops

if __name__ == '__main__':
    
    print("layer 12 768 2048")
    compute_transformer_flops(50257, 1024, 12, 768, 25, 2048)

    print("layer 24 1024 2752")
    compute_transformer_flops(50257, 1024, 24, 1024, 25, 2752)

    print("layer 36 1280 3392")
    compute_transformer_flops(50257, 1024, 36, 1280, 25, 3392)
    
    print("layer 48 1600 4288")
    compute_transformer_flops(50257, 1024, 48, 1600, 25, 4288)
    
    print("layer 48 1600 4288 with 16384 context length")
    compute_transformer_flops(50257, 16384, 48, 1600, 25, 4288)