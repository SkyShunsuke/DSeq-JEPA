import torch
import torch.nn as nn

class VICRegProjector(nn.Module):
    """MLP projector for VICReg loss
    params:
        in_dim: int, input feature dimension
        hidden_dim: int, hidden layer dimension
        out_dim: int, output feature dimension
        num_layers: int, number of layers in the MLP
        use_bn: bool, whether to use batch normalization
        activation: nn.Module, activation function to use
    """
    def __init__(self, in_dim: int, hidden_dim: int = 8192, out_dim: int = 8192, num_layers: int = 3 \
                , fc_bias: bool = False, **kwargs):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], bias=fc_bias))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
        
        # output layer
        layers.append(nn.Linear(dims[-2], dims[-1], bias=fc_bias))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        """Assume x is of shape (B, N, in_dim)"""
        B, N, _ = x.shape
        x = x.view(B * N, -1).contiguous()  # (B*N, in_dim)
        x = self.mlp(x)  # (B*N, out_dim)
        x = x.view(B, N, -1).contiguous()  # (B, N, out_dim)
        return x