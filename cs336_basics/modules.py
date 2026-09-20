import math
import torch

from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor


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
        self.weights = torch.empty(out_features, in_features)
        std = math.sqrt(2 / (self.in_features + self.out_features))
        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(self.weights, 0, std, -3*std, 3*std))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weights, x, "out in, ... in -> ... out")
        
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
        self.weights = torch.empty(self.num_embeddings, self.embedding_dim)
        self.weights = torch.nn.Parameter(torch.nn.init.trunc_normal_(self.weights, 0, 1, -3, 3))
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]
    
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
        self.weights = torch.nn.Parameter(torch.ones(self.d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape
        (batch_size, sequence_length, d_model) and return a tensor of the same shape
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weights
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
        self.w1_weight = torch.nn.Parameter(torch.ones(self.d_ff, self.d_model))
        self.w2_weight = torch.nn.Parameter(torch.ones(self.d_model, self.d_ff))
        self.w3_weight = torch.nn.Parameter(torch.ones(self.d_ff, self.d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_res = self.silu(einsum(self.w1_weight, x, "ff model, ... model -> ... ff"))
        w3_res = einsum(self.w3_weight, x, "ff model, ... model -> ... ff")
        return einsum(self.w2_weight, (silu_res * w3_res), "model ff, ... ff -> ... model")
    
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
        self.q_proj_weight = torch.nn.Parameter(torch.ones(self.d_model, self.d_model))
        self.k_proj_weight = torch.nn.Parameter(torch.ones(self.d_model, self.d_model))
        self.v_proj_weight = torch.nn.Parameter(torch.ones(self.d_model, self.d_model))
        self.o_proj_weight = torch.nn.Parameter(torch.ones(self.d_model, self.d_model))
        
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"], 
                rope: RoPE | None = None, token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> torch.Tensor:
        """
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        """
        seq_len = in_features.shape[-2]
        
        query = einsum(in_features, self.q_proj_weight, "... d, d_o d -> ... d_o") 
        query = rearrange(query, '... seq_len (num_head d_k) -> ... num_head seq_len d_k', 
                                num_head=self.num_heads)
        key = einsum(in_features, self.k_proj_weight, "... d, d_o d -> ... d_o")
        key = rearrange(key, '... seq_len (num_head d_k) -> ... num_head seq_len d_k', 
                                        num_head=self.num_heads)
        if rope:
            query = rope.forward(query, token_positions)
            key = rope.forward(key, token_positions)
        
        value = einsum(in_features, self.v_proj_weight, "... d, d_o d -> ... d_o")
        value = rearrange(value, '... seq_len (num_head d_k) -> ... num_head seq_len d_k', 
                                        num_head=self.num_heads)
        
        "scaled_dot_product_attention output shape would be ... num_head seq_len d_model/num_head"
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
        attention_res = scaled_dot_product_attention(query, key, value, mask)

        attention_res = rearrange(attention_res, '... num_head seq_len d_k -> ... seq_len (num_head d_k)')             
        return einsum(attention_res, self.o_proj_weight, "... d, d_out d -> ... d_out")
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, weights: dict[str, Tensor], rope: RoPE):
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
        self.rmsnorm_layer1 = RMSNorm(d_model, 1e-5)
        self.rmsnorm_layer1.load_state_dict({"weights": weights["ln1.weight"]})
        self.multihead_self_attention = MultiheadSelfAttention(d_model, num_heads)
        self.multihead_self_attention.load_state_dict({"q_proj_weight": weights["attn.q_proj.weight"], 
                                                        "k_proj_weight": weights["attn.k_proj.weight"],
                                                        "v_proj_weight": weights["attn.v_proj.weight"],
                                                        "o_proj_weight": weights["attn.output_proj.weight"]})
        self.rmsnorm_layer2 = RMSNorm(d_model, 1e-5)
        self.rmsnorm_layer2.load_state_dict({"weights": weights["ln2.weight"]})
        self.swiglu = SwiGLU(d_model, d_ff)
        self.swiglu.load_state_dict({"w1_weight": weights["ffn.w1.weight"], 
                                     "w2_weight": weights["ffn.w2.weight"], 
                                     "w3_weight": weights["ffn.w3.weight"]})
        
    def forward(self, in_features: Float[Tensor, " ... sequence_length d_model"], 
                    token_positions: Int[Tensor, " ... sequence_length"] | None = None):
        if not token_positions:
            token_positions = torch.arange(
                in_features.shape[-2],
                device=in_features.device,
            )
        first_half = in_features + \
                        self.multihead_self_attention.forward(
                            self.rmsnorm_layer1.forward(in_features), self.rope, token_positions)
        output = first_half + self.swiglu(self.rmsnorm_layer2.forward(first_half))
        return output

class TransformerLM(torch.nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor]):
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
        self.weights = weights
            
    def construct_weights(self, layer_id : int):
        transformer_block_weights = {}
        transformer_block_weights["ln1.weight"] = self.weights[f"layers.{layer_id}.ln1.weight"]
        transformer_block_weights["attn.q_proj.weight"] = self.weights[f"layers.{layer_id}.attn.q_proj.weight"]
        transformer_block_weights["attn.k_proj.weight"] = self.weights[f"layers.{layer_id}.attn.k_proj.weight"]
        transformer_block_weights["attn.v_proj.weight"] = self.weights[f"layers.{layer_id}.attn.v_proj.weight"]
        transformer_block_weights["attn.output_proj.weight"] = self.weights[f"layers.{layer_id}.attn.output_proj.weight"]
        transformer_block_weights["ln2.weight"] = self.weights[f"layers.{layer_id}.ln2.weight"]
        transformer_block_weights["ffn.w1.weight"] = self.weights[f"layers.{layer_id}.ffn.w1.weight"]
        transformer_block_weights["ffn.w2.weight"] = self.weights[f"layers.{layer_id}.ffn.w2.weight"]
        transformer_block_weights["ffn.w3.weight"] = self.weights[f"layers.{layer_id}.ffn.w3.weight"]
        return transformer_block_weights
    
    def forward(self, in_indices: Int[Tensor, " batch_size sequence_length"]) -> torch.Tensor:
        rope = RoPE(self.rope_theta, self.d_model // self.num_heads, self.context_length)
        
        "Get input embeddings. Tensor size is batch_size sequence_length d_model"
        token_embedding = Embedding(self.vocab_size, self.d_model)
        token_embedding.load_state_dict({"weights": self.weights["token_embeddings.weight"]})
        in_features = token_embedding.forward(in_indices)
        
        "Transformer Block"
        for i in range(self.num_layers):
            transformer_block_weights = self.construct_weights(i)
            in_features = TransformerBlock(self.d_model, self.num_heads, self.d_ff, 
                                            transformer_block_weights, rope).forward(in_features)
        
        "Norm"
        rmsnorm_final_layer = RMSNorm(self.d_model, 1e-5)
        rmsnorm_final_layer.load_state_dict({"weights": self.weights["ln_final.weight"]})
        norm_features = rmsnorm_final_layer.forward(in_features)
        
        "Linear"
        linear_layer = Linear(self.d_model, self.vocab_size)
        linear_layer.load_state_dict({"weights": self.weights["lm_head.weight"]})
        output_embedding = linear_layer.forward(norm_features)
        
        return output_embedding

