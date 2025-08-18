import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.cnn import *
from models.mlp import *
from models.transformer import * 

# ----------------------------------------------------------------
# create full GxE transformer for genomic prediction
class GxE_Transformer(nn.Module):

    """
    """
    
    def __init__(self,
                 dropout: float = 0.1,
                 hidden_dim: int = 768,
                 n_hidden: int =  4,
                 hidden_activation: nn.Module = nn.GELU(),
                 final_activation: nn.Module = nn.Identity(),
                 g_enc: bool = True,
                 e_enc: bool = True,
                 config = None
                 ):
        super().__init__()

        # set attributes
        self.g_enc_flag, self.e_enc_flag = g_enc, e_enc
        if g_enc:
            self.g_encoder = G_Encoder(config)
        if e_enc:
            self.e_encoder = E_Encoder(output_dim=config.n_embd)
        self.hidden_dim = config.n_embd
        self.hidden_layers = nn.ModuleList(
            [Block(self.g_encoder.output_dim if i == 0 else hidden_dim,
                   hidden_dim,
                   dropout=dropout,
                   activation=hidden_activation,
                   ) for i in range(n_hidden)]
        )
        
        # init final layer (output of 1 for regression)
        self.final_layer = nn.Linear(hidden_dim, 1) # CAN CHANGE INPUT, OUTPUT SIZE FOR LAYERS

    def forward(self, x):

        # only pass through G, E encoders if they exist
        if self.g_enc_flag and self.e_enc_flag:
            g_enc = self.g_encoder(x["g_data"]).mean(dim=1) # (B, T, n_embd) --> (B, n_embd)
            e_enc = self.e_encoder(x["e_data"])
            assert g_enc.shape == e_enc.shape, "G and E encoders must output same shape"
            x = g_enc + e_enc
        elif self.g_enc_flag:
            x = self.g_encoder(x["g_data"]).mean(dim=1)
        else:
            x = self.e_encoder(x["e_data"])
        for layer in self.hidden_layers:
            x = x + layer(x)

        return self.final_layer(x)

# ----------------------------------------------------------------
# create full GxE transformer (two transformer blocks) for genomic prediction
class GxE_FullTransformer(nn.Module):

    """
    Full transformer for genomic prediction
    """
    
    def __init__(self,
                 g_enc: bool = True,
                 e_enc: bool = True,
                 config = None
                 ):
        super().__init__()

        # set attributes
        self.g_enc_flag, self.e_enc_flag = g_enc, e_enc
        if g_enc:
            self.g_encoder = G_Encoder(config)
        if e_enc:
            self.e_encoder = E_Encoder(output_dim=config.n_embd)
        self.hidden_dim = config.n_embd
        self.hidden_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
            + [nn.LayerNorm(config.n_embd)]
        )
        
        # init final layer (output of 1 for regression)
        self.final_layer = nn.Linear(config.n_embd, 1)

    def forward(self, x):

        # only pass through G, E encoders if they exist
        if self.g_enc_flag and self.e_enc_flag:
            g_enc = self.g_encoder(x["g_data"])
            e_enc = self.e_encoder(x["e_data"]).unsqueeze(dim=1)
            # assert g_enc.shape == e_enc.shape, "G and E encoders must output same shape"
            x = g_enc + e_enc
        elif self.g_enc_flag:
            x = self.g_encoder(x["g_data"])
        else:
            x = self.e_encoder(x["e_data"])
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.mean(dim=1) # (B, T, n_embd) -> (B, n_embd)

        return self.final_layer(x)

class GxE_LD_FullTransformer(nn.Module):
    """
    Full transformer for genomic and environmental data with an added LD-encoding CNN
    """

    def __init__(self,):
    
    def forward(self, x):