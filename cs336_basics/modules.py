import math
import numpy.typing as npt
import os
import torch

from collections.abc import Callable, Iterable
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor
from typing import Optional, IO, BinaryIO



def softmax(x: torch.Tensor, i_d: int) -> torch.Tensor:
    """
    Apply softmax to the 𝑖-th dimension of the
    input tensor. The output tensor should have the same shape as the input tensor, but its 𝑖-th
    dimension will now have a normalized probability distribution. Use the trick of subtracting the
    maximum value in the 𝑖-th dimension from all elements of the 𝑖-th dimension to avoid numerical
    stability issues.
    """
    x_reduce_by_max = x - torch.max(x, dim=i_d, keepdim=True).values
    x_power_e = torch.exp(x_reduce_by_max)
    return x_power_e / torch.sum(x_power_e, dim=i_d, keepdim=True)

def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"]) -> Float[Tensor, " ... queries d_v"]:
    """
    Implement the scaled dot-product attention function. Your implementation should
    handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape
    (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like
    dimensions (if provided). The implementation should return an output with the shape
    (batch_size, ..., seq_len, d_v). See Section 3.2 for a discussion on batch-like dimensions.
    Your implementation should also support an optional user-provided boolean mask of shape
    (seq_len, seq_len). The attention probabilities of positions with a mask value of True should
    collectively sum to 1, and the attention probabilities of positions with a mask value of False
    should be zero.
    """
    score = einsum(Q, K, "... q d_k, ... k d_k -> ... q k").masked_fill(~mask, float("-inf"))
    score = softmax(score / math.sqrt(K.shape[-1]), -1)
    return einsum(score, V, "... q k, ... k d_v -> ... q d_v")

def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], 
                  targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """
    compute the cross-entropy loss, which takes in predicted logits
    (oi) and targets (xi+1) and computes the cross-entropy li= -log softmax(oi)[xi+1]
    """
    targets = targets.unsqueeze(-1)
    max_element = torch.max(inputs, dim=-1, keepdim=True).values
    selected_logits = torch.gather(inputs, dim=-1, index=targets) - max_element
    inputs_reduce_by_max = torch.exp(inputs - max_element)
    processed_selected = torch.log(torch.sum(inputs_reduce_by_max, dim=-1, keepdim=True)) - selected_logits
    return torch.sum(processed_selected) / torch.numel(processed_selected)

def lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float,
                        warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    ratio = 1 + math.cos((it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters))
    lr = min_learning_rate + ratio / 2 * (max_learning_rate - min_learning_rate)
    return lr
    
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters_list = list(parameters)
    l2_sum = 0
    for p in parameters_list:
        if p.grad is None:
            continue
        l2_sum += (p.grad ** 2).sum().item()
    l2_norm = math.sqrt(l2_sum)
    if l2_norm > max_l2_norm:
        for p in parameters_list:
            if p.grad is None:
                continue
            p.grad *= max_l2_norm / (l2_norm + 1e-6)
    return

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, 
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_size = len(dataset)
    input_start = torch.randint(low=0, high=(dataset_size - context_length), size=(batch_size,)).unsqueeze(1)
    label_start = input_start + 1
    offset = torch.arange(context_length).unsqueeze(0)
    input_index = input_start + offset
    label_index = label_start + offset
    dataset_tensor = torch.from_numpy(dataset)
    input_tensor = dataset_tensor[input_index].to(device)
    label_tensor = dataset_tensor[label_index].to(device)
    return (input_tensor, label_tensor)

def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    ckpt = {}
    ckpt["weights"] = model.state_dict()
    ckpt["optimizer_state"] = optimizer.state_dict()
    ckpt["iteration"] = iteration
    torch.save(ckpt, out)

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["weights"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["iteration"]

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        in_features: int  final dimension of the input
        out_features: int  final dimension of the output
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype 
        self.weight = torch.empty(out_features, in_features)
        std = math.sqrt(2 / (self.in_features + self.out_features))
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(self.weight, 0, std, -3*std, 3*std))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "out in, ... in -> ... out")
        
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int  Size of the vocabulary
        embedding_dim: int  Dimension of the embedding vectors, i.e., dmodel
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype 
        self.weight = torch.empty(self.num_embeddings, self.embedding_dim)
        self.weight = torch.nn.Parameter(torch.nn.init.trunc_normal_(self.weight, 0, 1, -3, 3))
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class RMSNorm(torch.nn.Module):
    "Root Mean Square Layer Normalization"
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int  Hidden dimension of the model
        eps: float = 1e-5  Epsilon value for numerical stability
        device: torch.device | None = None  Device to store the parameters on
        dtype: torch.dtype | None = None  Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype 
        self.weight = torch.nn.Parameter(torch.ones(self.d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)
    
class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if device:
            self.device = device
        if dtype:
            self.dtype = dtype 
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_res = self.silu(self.w1.forward(x))
        w3_res = self.w3.forward(x)
        output = self.w2.forward(silu_res * w3_res)
        return output
    
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):

        """
        Construct the RoPE module and create buffers if needed.
        Args:
            theta: float  Θ value for the RoPE
            d_k: int  dimension of query and key vectors
            max_seq_len: int  Maximum sequence length that will be input
            device: torch.device | None = None  Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        if device:
            self.device = device
        rotation_d = int(self.d_k / 2)
        index = 2 * torch.arange(rotation_d) / self.d_k
        inv_freq = 1 / torch.pow(self.theta, index)
        position = torch.arange(self.max_seq_len)
        self.sin_matrix = torch.sin(einsum(position, inv_freq, "seq_len, rotation_d -> seq_len rotation_d"))
        self.cos_matrix = torch.cos(einsum(position, inv_freq, "seq_len, rotation_d -> seq_len rotation_d"))
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note
        that you should tolerate 𝑥 with an arbitrary number of batch dimensions. You should assume
        that the token positions are a tensor of shape (..., seq_len) specifying the token positions of
        𝑥 along the sequence dimension.
        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
        the sequence dimension.
        """
        
        "Tensor size is ..., seq_len, d_k / 2, 2"
        reshaped_x = x.reshape((*x.shape[:-1], self.d_k // 2, 2))
        
        "Tensor size is ..., seq_len, d_k / 2"
        selected_sin = self.sin_matrix[token_positions]
        selected_cos = self.cos_matrix[token_positions] 
        x0 = reshaped_x[..., 0] * selected_cos - reshaped_x[..., 1] * selected_sin       
        x1 = reshaped_x[..., 0]* selected_sin + reshaped_x[..., 1] * selected_cos
        res = torch.stack([x0, x1], dim=-1)
        return res.reshape(*res.shape[:-2], self.d_k)

class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(self.d_model, self.d_model)
        self.k_proj = Linear(self.d_model, self.d_model)
        self.v_proj = Linear(self.d_model, self.d_model)
        self.output_proj = Linear(self.d_model, self.d_model)
        
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"], 
                rope: RoPE | None = None, token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> torch.Tensor:
        """
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        """
        seq_len = in_features.shape[-2]
        
        query = self.q_proj.forward(in_features)
        query = rearrange(query, '... seq_len (num_head d_k) -> ... num_head seq_len d_k', 
                                num_head=self.num_heads)
        key = self.k_proj.forward(in_features)
        key = rearrange(key, '... seq_len (num_head d_k) -> ... num_head seq_len d_k', 
                                        num_head=self.num_heads)
        if rope:
            query = rope.forward(query, token_positions)
            key = rope.forward(key, token_positions)
        
        value = self.v_proj.forward(in_features)
        value = rearrange(value, '... seq_len (num_head d_k) -> ... num_head seq_len d_k', 
                                        num_head=self.num_heads)
        
        "scaled_dot_product_attention output shape would be ... num_head seq_len d_model/num_head"
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
        attention_res = scaled_dot_product_attention(query, key, value, mask)

        attention_res = rearrange(attention_res, '... num_head seq_len d_k -> ... seq_len (num_head d_k)') 
        output = self.output_proj.forward(attention_res)            
        return output
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RoPE):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.ln1 = RMSNorm(d_model, 1e-5)
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model, 1e-5)
        self.ffn = SwiGLU(d_model, d_ff)
        
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"], 
                    token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        if not token_positions:
            token_positions = torch.arange(
                in_features.shape[-2],
                device=in_features.device,
            )
        first_half = in_features + \
                        self.attn.forward(self.ln1.forward(in_features), self.rope, token_positions)
        output = first_half + self.ffn(self.ln2.forward(first_half))
        return output

class TransformerLM(torch.nn.Module):
    def __init__(self, 
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.token_embeddings = Embedding(self.vocab_size, self.d_model)
        self.rope = RoPE(self.rope_theta, self.d_model // self.num_heads, self.context_length)
        self.layers = torch.nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.rope))
        self.ln_final = RMSNorm(self.d_model, 1e-5)
        self.lm_head = Linear(self.d_model, self.vocab_size)
            
    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> torch.Tensor:
        "Get input embeddings. Tensor size is batch_size sequence_length d_model"
        in_features = self.token_embeddings.forward(in_indices)
        
        "Transformer Block"
        for i in range(self.num_layers):
            in_features = self.layers[i].forward(in_features)
        
        "Norm"
        norm_features = self.ln_final.forward(in_features)
        
        "Linear"
        output_embedding = self.lm_head.forward(norm_features)
        
        return output_embedding

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        defaults = {"alpha": lr, "beta1": betas[0], "beta2": betas[1], "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["alpha"] # Get the learning rate.
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                t = state["t"]
                m = state["m"]
                v = state["v"]
                alpha_t = alpha * math.sqrt(1 - pow(beta2, t)) / (1 - pow(beta1, t))
                p.data -= alpha * weight_decay * p.data
                m = beta1 * m + (1 - beta1) * p.grad
                v = beta2 * v + (1 - beta2) * pow(p.grad, 2)
                p.data -= alpha_t * m / (v.sqrt() + eps)
                state["t"] = t + 1 # Increment iteration number.
                state["m"] = m
                state["v"] = v
        return loss