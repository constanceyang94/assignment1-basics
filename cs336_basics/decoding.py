from cs336_basics.modules import * 

# To be changed
end_token = 0

def decoding(transformer_lm: TransformerLM, 
             prompt: Int[Tensor, "prompt_length"],
             max_tokens: int = 100,
             temperature: float = 1,
             top_p: float | None = None):
    response = torch.tensor(0, dtype=torch.long)
    next_token = None
    while len(response) < max_tokens:
        output = transformer_lm.forward(prompt)
        if temperature == 0:
            next_token = torch.argmax(output, -1)
        else:
            probablity = softmax(output / temperature, -1)
            # Implement top_p
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probablity, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative_probs - sorted_probs >= top_p
                sorted_probs[remove] = 0.0
                filtered_probs = torch.zeros_like(probablity)
                filtered_probs.scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_probs
                )
                probablity = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(probablity, num_samples=1)
        prompt = torch.cat([prompt, next_token])
        response = torch.cat([response, next_token])
        if next_token.item() == end_token:
            break
    return response