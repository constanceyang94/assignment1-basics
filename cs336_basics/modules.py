import math
import torch

from einops import einsum


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