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
                 gxe_enc: bool = True,
                 moe: bool = True,
                 config = None
                 ):
        super().__init__()

        self.config = config

        # set attributes
        self.g_encoder = G_Encoder(config) if g_enc else None
        self.e_encoder = E_Encoder(output_dim=config.n_embd,
                                   n_hidden=config.n_mlp_layer,
                                   dropout=config.dropout) if e_enc else None
        self.ld_encoder = LD_Encoder(input_dim=config.vocab_size,
                                     output_dim=config.n_embd,
                                     num_blocks=config.n_ld_layer,
                                     dropout=config.dropout) if ld_enc else None
        self.moe_w = nn.Parameter(torch.tensor([1.0, 1.0, 1.0])) if moe else None
        if gxe_enc == "tf":
            self.gxe_enc = "tf"
            self.hidden_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_gxe_layer)]
            + [nn.LayerNorm(config.n_embd)]
            )
        elif gxe_enc == "mlp":
            self.gxe_enc = "mlp"
            self.hidden_layers = nn.ModuleList(
                [Block(self.g_encoder.output_dim if i == 0 else config.n_embd,
                    config.n_embd,
                    dropout=config.dropout,
                    activation=nn.GELU(),
                    ) for i in range(config.n_gxe_layer)]
            )
        elif gxe_enc == "cnn":
            self.gxe_enc = "cnn"
            # write cnn code here
        else:
            raise ValueError("gxe_enc must be one of ['tf', 'mlp', 'cnn']")

        
        # init final layer (output of 1 for regression)
        self.final_layer = nn.Linear(config.n_embd, 1) # CAN CHANGE INPUT, OUTPUT SIZE FOR LAYERS
    
    def _concat(self, g_enc, e_enc, ld_enc):
        if self.moe_w is not None:
            g_enc = self.moe_w[0] * g_enc
            e_enc = self.moe_w[1] * e_enc
            ld_enc = self.moe_w[2] * ld_enc
        return g_enc + e_enc + ld_enc

    def _forward_tf(self, g_enc, e_enc, ld_enc):
        if isinstance(e_enc, torch.Tensor): 
            e_enc = e_enc.unsqueeze(dim=1)
        x = self._concat(g_enc, e_enc, ld_enc)
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.mean(dim=1) # (B, T, n_embd) -> (B, n_embd)
        return x

    def _forward_mlp(self, g_enc, e_enc, ld_enc):
        # convert [B, T, C] -> [B, C]
        if isinstance(g_enc, torch.Tensor):
            g_enc = g_enc.mean(dim=1)
        if isinstance(ld_enc, torch.Tensor):
            ld_enc = ld_enc.mean(dim=1)
        x = self._concat(g_enc, e_enc, ld_enc)
        for layer in self.hidden_layers:
            x = x + layer(x)
        return x
    
    def _forward_cnn(self, g_enc, e_enc, ld_enc):
        if isinstance(e_enc, torch.Tensor): 
            e_enc = e_enc.unsqueeze(dim=1)
        x = self._concat(g_enc, e_enc, ld_enc)
        for layer in self.hidden_layers:
            x = x + layer(x)
        x = x.mean(dim=1) # (B, T, n_embd) -> (B, n_embd) 
        return x
    
    def forward(self, x):
        g_enc = self.g_encoder(x["g_data"]) if self.g_encoder else 0
        e_enc = self.e_encoder(x["e_data"]) if self.e_encoder else 0
        ld_enc = 0
        if self.ld_encoder:
            ld_feats = F.one_hot(x["g_data"].long(), num_classes=self.ld_encoder.input_dim)
            ld_enc = self.ld_encoder(ld_feats.float())
        if self.gxe_enc == "tf":
            x = self._forward_tf(g_enc, e_enc, ld_enc)
        elif self.gxe_enc == "mlp":
            x = self._forward_mlp(g_enc, e_enc, ld_enc)
        elif self.gxe_enc == "cnn":
            x = self._forward_cnn(g_enc, e_enc, ld_enc)
        else:
            raise ValueError("gxe_enc must be one of ['tf', 'mlp', 'cnn']")

        return self.final_layer(x)