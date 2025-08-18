import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    model_type: str = "base"
    block_size: int = 2240 # max sequence length
    vocab_size: int = 3 # 0, 0.5, 1
    n_layer: int = 4 # number of transformer layers
    n_head: int = 8 # number of heads
    n_embd: int = 768 # embedding size