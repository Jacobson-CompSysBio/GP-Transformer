import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# ----------------------------------------------------------------
# CNN module for LD-based features
class ResNetBlock1D(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 n_convs: int = 2,
                 kernel_size: int = 3,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 dropout: float = 0.25,
                 use_batchnorm: bool = True):

        super().__init__()
        assert n_convs >= 1, "n_convs must be at least 1"

        padding = kernel_size // 2 

        layers = []
        for i in range(n_convs):
            conv_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv1d(conv_in, out_channels,
                                    kernel_size,
                                    padding=padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))
            
            layers.append(activation)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.body = nn.Sequential(*layers)
        self.proj = (
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=1)
                      if in_channels != out_channels
                      else nn.Identity()
        )

    def forward(self, x):
        # x: (B, C, L)
        y = self.body(x)
        res = self.proj(x)
        return y + res
    
class LD_Encoder(nn.Module):
    """
    CNN-based encoder for LD-based features
    input: (B, L, C) 
    output: (B, L, emb_dim)
    """

    def __init__(self,
                 input_dim: int = 3,
                 output_dim: int = 768,
                 num_blocks: int = 4,
                 n_convs_per_block: int = 2,
                 kernel_size: int = 3,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 dropout: float = 0.25):
        
        super().__init__()
        self.input_dim = input_dim

        # stem to reach output dim
        self.stem = nn.Conv1d(input_dim, output_dim, kernel_size=1)

        self.blocks = nn.ModuleList(
            [
                ResNetBlock1D(
                    in_channels=output_dim,
                    out_channels=output_dim,
                    n_convs=n_convs_per_block,
                    kernel_size=kernel_size,
                    activation=activation,
                    dropout=dropout,
                    use_batchnorm=True
                ) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        # accept B, T, C and transpose to B, C, T
        x = x.transpose(1, 2) # B, T, C --> B, C, T

        # project up to emb_dim    
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        
        # transpose back to original shape
        x = x.transpose(1, 2) # B, C, T --> B, T, C
        return x

class GxE_CNN(nn.Module):
    pass
