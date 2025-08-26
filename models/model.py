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
                 g_enc: bool = True,
                 e_enc: bool = True,
                 ld_enc: bool = True,
                 final_tf: bool = True,
                 config = None
                 ):
        super().__init__()

        # set attributes
        self.g_encoder = G_Encoder(config) if g_enc else None
        self.e_encoder = E_Encoder(output_dim=config.n_embd, dropout=config.dropout) if e_enc else None
        self.ld_encoder = LD_Encoder(input_dim=config.vocab_size,
                                         output_dim=config.n_embd,
                                         num_blocks=config.n_layer,
                                         dropout=config.dropout) if ld_enc else None
        if final_tf:
            self.final_tf = True
            self.hidden_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
            + [nn.LayerNorm(config.n_embd)]
            )
        else:
            self.final_tf = False
            self.hidden_layers = nn.ModuleList(
                [Block(self.g_encoder.output_dim if i == 0 else config.n_embd,
                    config.n_embd,
                    dropout=config.dropout,
                    activation=nn.GELU(),
                    ) for i in range(config.n_layer)]
            )
        
        # init final layer (output of 1 for regression)
        self.final_layer = nn.Linear(config.n_embd, 1) # CAN CHANGE INPUT, OUTPUT SIZE FOR LAYERS

    def _forward_tf(self, x, g_enc, e_enc, ld_enc):
        e_enc = e_enc.unsqueeze(dim=1)
        x = g_enc + e_enc + ld_enc
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.mean(dim=1) # (B, T, n_embd) -> (B, n_embd)
        return x

    def _forward_mlp(self, x, g_enc, e_enc, ld_enc):
        g_enc = g_enc.mean(dim=1)
        x = g_enc + e_enc + ld_enc
        for layer in self.hidden_layers:
            x = x + layer(x)
        return x
    
    def forward(self, x):

        # only pass through G, E encoders if they exist
        g_enc = self.g_encoder(x["g_data"]) if self.g_encoder else 0
        e_enc = self.e_encoder(x["e_data"]) if self.e_encoder else 0
        ld_enc = 0
        if self.ld_encoder:
            ld_feats = F.one_hot(x["g_data"].long(), num_classes=self.ld_encoder.input_dim)
            ld_enc = self.ld_encoder(ld_feats.float())
        if self.final_tf:
            x = self._forward_tf(x, g_enc, e_enc, ld_enc)
        else:
            x = self._forward_mlp(x, g_enc, e_enc, ld_enc)

        return self.final_layer(x)

