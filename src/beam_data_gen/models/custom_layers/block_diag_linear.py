import torch
import torch.nn as nn


class LinearBlockDiag(nn.Module):
    def __init__(self, N: int, in_features: int, out_features: int):
        """Creates a linear weights packed along the block diagonal 
           of the weight 

        Args:
            N (int): Number of independent weights to stack along the block
            in_features (int): number of features in the input
            out_features (int): number of features in the output
        """
        super().__init__()
        
        # weights = list(nn.Parameter(torch.randn(out_features, in_features)) for _ in range(N))
        self.weight = nn.Parameter(torch.randn(N * out_features, N * in_features))
        self.bias = nn.Parameter(torch.randn(N * out_features))
        
        learnable_mask = torch.kron(torch.eye(N), torch.ones(out_features, in_features))
        # Learnable mask: 1 where learnable, 0 where constant
        self.register_buffer("mask", learnable_mask.clone().float())

        # Make learnable part random init
        learnable_init = torch.randn(N * out_features, N * in_features)
        with torch.no_grad():
            self.weight.data = self.mask * learnable_init

    def forward(self, input: torch.tensor):
        full_weight = self.mask * self.weight
        return input @ full_weight.T + self.bias
