import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    block_size: int = 2240 # max sequence length
    vocab_size: int = 3 # 0, 0.5, 1
    n_g_layer: int = 1 # number of g_tf layers
    n_ld_layer: int = 1 # number of ld layers
    n_mlp_layer: int = 1 # num mlp layers
    n_gxe_layer: int = 4 # number of gxe layers
    n_head: int = 8 # number of heads
    n_embd: int = 768 # embedding size
    dropout: float = 0.25 # dropout frequency