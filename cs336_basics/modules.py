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