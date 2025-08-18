import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ----------------------------------------------------------------
# CNN module for LD-based features
class ResNetBlock1D(nn.Module):
    
    def __init__(self,
                 input_channels = 768,
                 output_channels = 768,
                 num_conv = 2,
                 activation = nn.ReLU(),
                 dropout = 0.1):

        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout) 
        self.convs = nn.ModuleList(
           [nn.Conv1d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      padding=1) for _ in range(num_conv)]
        )

        if input_channels != output_channels:
            self.residual_conv = nn.Conv1d(in_channels=input_channels,
                                           out_channels=output_channels,
                                           kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        res = self.residual_conv(x) if self.residual_conv else x
        for conv in self.convs:
            x = res + self.dropout(self.activation(conv(x)))
        return x
    
class LD_Encoder(nn.Module):
    """
    CNN-based encoder for LD-based features 
    """

    def __init__(self,
                 input_dim: int = 3,
                 output_dim: int = 768,
                 num_blocks: int = 4,
                 kernel_size: int = 3,
                 activation: nn.Module = nn.ReLU(),
                 dropout: float = 0.1):
        
        super().__init__()
        self.input_block = ResNetBlock1D(input_channels=input_dim,
                                          output_channels=output_dim,
                                          num_conv=num_blocks,
                                          activation=activation,
                                          dropout=dropout)
        self.res_blocks = nn.ModuleList(
            [self.input_block] +
            [
                ResNetBlock1D(input_channels=output_dim,
                              output_channels=output_dim,
                              num_conv=num_blocks,
                              activation=activation,
                              dropout=dropout) for _ in range(num_blocks)
            ]
        )
    
    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        return x
