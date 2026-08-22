import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeightMLP(nn.Module):
    """
    A small Multi-Layer Perceptron (MLP) that maps a 13-dimensional feature representation
    to 2 normalized weights (w_colour, w_texture) using Softmax activation.
    """
    def __init__(self, input_dim=13):
        super().__init__()
        # 13 -> 16 -> 8 -> 2 hidden architecture suitable for a small dataset
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, 13).
            
        Returns:
            torch.Tensor: Normalized weights of shape (batch, 2), where:
                          weights[:, 0] = w_colour
                          weights[:, 1] = w_texture
        """
        h1 = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        h2 = F.leaky_relu(self.fc2(h1), negative_slope=0.1)
        logits = self.fc3(h2)
        
        # Softmax along the last dimension to get weights that sum to 1.0
        weights = F.softmax(logits, dim=-1)
        return weights
