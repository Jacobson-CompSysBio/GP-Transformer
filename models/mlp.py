import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
# ----------------------------------------------------------------
# create MLP block for flat layers
class Block(nn.Module):
    """
    Flat block with a single linear layer, layernorm, dropout and activation
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.25,
                 layernorm: bool = True,
                 ):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if layernorm:
            self.layernorm = nn.LayerNorm(output_dim)

    # fwd pass
    def forward(self, x):
        x = self.dropout(self.activation(self.fc(x)))
        if hasattr(self, 'layernorm'):
            x = self.layernorm(x)
        
        return x

# create MLP encoder for environmental covariates
class E_Encoder(nn.Module):

    def __init__(self,
                 input_dim: int = 374,
                 output_dim: int = 128,
                 hidden_dim: int = 128,
                 n_hidden: int = 2,
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.25,
                 layernorm: bool = True,
                 ):
        super().__init__()

        layers=[]
        for i in range(n_hidden):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(Block(in_dim, hidden_dim, activation, dropout, layernorm))
        self.hidden_layers = nn.ModuleList(layers)

        # add final layer
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.final_activation = activation

    # forward pass
    def forward(self, x):
        # through hidden layers
        for i, layer in enumerate(self.hidden_layers):
            # can't use residual connection for first layer, since input since doesn't match hidden layer sizes
            if i == 0:
                x = layer(x)
            else: 
                x = x + layer(x)

        # through final layer
        x = self.final_activation(self.final_layer(x))

        return x