import math
import torch

from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor


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
        print("x shape is " + str(x.shape))
        reshaped_x = x.reshape((*x.shape[:-1], self.d_k // 2, 2))
        
        "Tensor size is ..., seq_len, d_k / 2"
        selected_sin = self.sin_matrix[token_positions]
        selected_cos = self.cos_matrix[token_positions] 
        x0 = reshaped_x[..., 0] * selected_cos - reshaped_x[..., 1] * selected_sin       
        x1 = reshaped_x[..., 0]* selected_sin + reshaped_x[..., 1] * selected_cos
        res = torch.stack([x0, x1], dim=-1)
        return res.reshape(*res.shape[:-2], self.d_k)
    
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