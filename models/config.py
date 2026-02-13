import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    block_size: int = 2024 # max sequence length
    vocab_size: int = 3 # 0, 0.5, 1
    g_input_type: str = "tokens" # genotype input representation: "tokens" or "grm"
    n_g_layer: int = 1 # number of g_tf layers
    n_ld_layer: int = 1 # number of ld layers
    n_mlp_layer: int = 1 # num mlp layers
    n_gxe_layer: int = 4 # number of gxe layers
    n_head: int = 8 # number of heads
    n_embd: int = 768 # embedding size
    dropout: float = 0.25 # dropout frequency
    n_env_fts: int = 705 # number of environmental features
    # Optional metadata for richer environment tokenization in FullTransformer
    env_stage_ids: Optional[List[int]] = None
    n_env_stages: int = 1
    n_env_categorical: int = 0
    env_cat_cardinalities: Optional[List[int]] = None
    env_feature_id_emb: bool = False
    env_stage_id_emb: bool = False
    env_cat_embeddings: bool = False
