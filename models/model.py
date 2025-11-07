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
                 gxe_enc: str = "tf",
                 moe: bool = True,
                 config = None
                 ):
        super().__init__()

        self.config = config

        # set attributes
        self.g_encoder = G_Encoder(config) if g_enc else None
        self.e_encoder = E_Encoder(input_dim=config.n_env_fts,
                                   output_dim=config.n_embd,
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
            self.hidden_layers = nn.ModuleList(
                [ResNetBlock1D(in_channels=self.g_encoder.output_dim if i == 0 else config.n_embd,
                               out_channels=config.n_embd,
                               dropout=config.dropout,
                               use_batchnorm=True) for i in range(config.n_gxe_layer)]
            )
        else:
            raise ValueError("gxe_enc must be one of ['tf', 'mlp', 'cnn']")

        # init final layer
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
        #unsqueeze e to concat properly
        if isinstance(e_enc, torch.Tensor): 
            e_enc = e_enc.unsqueeze(dim=1)
        x = self._concat(g_enc, e_enc, ld_enc)

        x = x.transpose(1, 2) # (B, T, C) -> (B, C, T)
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.transpose(1, 2) # (B, C, T) -> (B, T, C)
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
    
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        g_trainable_params = sum(p.numel() for p in self.g_encoder.parameters() if p.requires_grad) if self.g_encoder else 0
        e_trainable_params = sum(p.numel() for p in self.e_encoder.parameters() if p.requires_grad) if self.e_encoder else 0
        ld_trainable_params = sum(p.numel() for p in self.ld_encoder.parameters() if p.requires_grad) if self.ld_encoder else 0
        gxe_trainable_params = trainable_params - g_trainable_params - e_trainable_params - ld_trainable_params
        print(f"Trainable parameters: {trainable_params:,}"
              f" (G: {g_trainable_params:,}, E: {e_trainable_params:,}, LD: {ld_trainable_params:,}, GxE: {gxe_trainable_params:,})")


# create full GxE transformer for genomic prediction
class GxE_ResidualTransformer(nn.Module):

    """
    """
    
    def __init__(self,
                 g_enc: bool = True,
                 e_enc: bool = True,
                 ld_enc: bool = True,
                 gxe_enc: str = "tf",
                 moe: bool = True,
                 residual: bool = False,
                 config = None
                 ):
        super().__init__()

        self.config = config
        self.residual = residual

        # set attributes
        self.g_encoder = G_Encoder(config) if g_enc else None
        self.e_encoder = E_Encoder(input_dim=config.n_env_fts,
                                   output_dim=config.n_embd,
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
            self.hidden_layers = nn.ModuleList(
                [ResNetBlock1D(in_channels=self.g_encoder.output_dim if i == 0 else config.n_embd,
                               out_channels=config.n_embd,
                               dropout=config.dropout,
                               use_batchnorm=True) for i in range(config.n_gxe_layer)]
            )
        else:
            raise ValueError("gxe_enc must be one of ['tf', 'mlp', 'cnn']")

        # env head to predict per-year mean yield (if residual)
        self.ymean_head = nn.Linear(config.n_embd, 1) if self.e_encoder is not None else None

        # init final layer --> predicts residual in residual mode, total otherwise
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
        #unsqueeze e to concat properly
        if isinstance(e_enc, torch.Tensor): 
            e_enc = e_enc.unsqueeze(dim=1)
        x = self._concat(g_enc, e_enc, ld_enc)

        x = x.transpose(1, 2) # (B, T, C) -> (B, C, T)
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.transpose(1, 2) # (B, C, T) -> (B, T, C)
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

        # non-residual mode: final layer predicts total yield 
        if not self.residual:
            return self.final_layer(x)
        
        # residual mode:
        ymean_pred = self.ymean_head(e_enc) if self.ymean_head is not None else 0
        resid_pred = self.final_layer(x)

        # option: detach ymean_pred to prevent env head from learning residual signal
        if self.detach_ymean_in_sum and isinstance(ymean_pred, torch.Tensor):
            total_pred = ymean_pred.detach() + resid_pred
        else:
            total_pred = ymean_pred + resid_pred
        return {'total': total_pred,
                'ymean': ymean_pred,
                'resid': resid_pred,
        }
    
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        g_trainable_params = sum(p.numel() for p in self.g_encoder.parameters() if p.requires_grad) if self.g_encoder else 0
        e_trainable_params = sum(p.numel() for p in self.e_encoder.parameters() if p.requires_grad) if self.e_encoder else 0
        ld_trainable_params = sum(p.numel() for p in self.ld_encoder.parameters() if p.requires_grad) if self.ld_encoder else 0
        gxe_trainable_params = trainable_params - g_trainable_params - e_trainable_params - ld_trainable_params
        print(f"Trainable parameters: {trainable_params:,}"
              f" (G: {g_trainable_params:,}, E: {e_trainable_params:,}, LD: {ld_trainable_params:,}, GxE: {gxe_trainable_params:,})")
    
