import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# look at Chinchilla for parameter recommendations
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
                 dropout: float = 0.1,
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

# ----------------------------------------------------------------
# config for transformer
@dataclass
class TransformerConfig:
    block_size: int = 2240 # max sequence length
    vocab_size: int = 3 # 0, 0.5, 1
    n_layer: int = 2 # number of transformer layers
    n_head: int = 8 # number of heads
    n_embd: int = 768 # embedding size

# positional encoding (sine/cosine)
class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super().__init__()
        # matrix of shape (seq_len, d_model)
        pe = torch.zeros(config.block_size, config.n_embd)

        # vector of shape (seq_len)
        position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)

        # vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, config.n_embd, 2).float() * (-math.log(10_000.0) / config.n_embd))

        # apply sin to even idxs, cos to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add batch dimension to pos encoding (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        """
        Parameters:
            x (torch.tensor): shape (batch_size, seq_len, embedding_dim)
        """
        x = x + (self.pe[:, :x.shape[1], :]).to(x.dtype) # (batch, seq_len, d_model)
        return self.dropout(x)


# transformer
class SelfAttention(nn.Module):
    """
    Bidirectional self attention with FlashAttention
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "hidden_dim must be divisible by head_size"

        # get attributes
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # init k, q, v linear layers (all in same layer)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        """
        calculate query, key, and value for all heads in batch and move head forward to be batched
        nh is "number of heads", hs is "head size", and C (# channels) = nh * hs
        e.g. GPT2: nh = 12, hs = 64, so nh * hs = C = 768 channels
        """

        # for this project, B = batch size, T = 2024 (sequence length), and C = 768 (hidden_dim)
        B, T, C = x.shape

        # get q, k, v
        qkv = self.c_attn(x) # (B, T, C * 3)

        # split and transpose for faster computation
        # pytorch process B and num_heads dim as batches, processes in parallel
        q, k, v = qkv.chunk(3, dim=2) # split along channels 
        head = C // self.n_head
        k = k.view(B, T, self.n_head, head).transpose(1, 2) # (B, num_heads, T, head_size)
        q = q.view(B, T, self.n_head, head).transpose(1, 2) # (B, num_heads, T, head_size)
        v = v.view(B, T, self.n_head, head).transpose(1, 2) # (B, num_heads, T, head_size)

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # no mask, so is_causal=False
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble head outputs side-by-side

        # output projection
        return self.c_proj(y)

class TransformerMLP(nn.Sequential):
    """
    MLP for transformer
    """
    def __init__(self, config):
        super().__init__(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

class TransformerBlock(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = TransformerMLP(config)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # resid connections
        x = x + self.dropout(self.attn(self.ln_1(x)))
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        return x

# create transformer encoder for genotype data
class G_Encoder(nn.Module):

    def __init__(self, config
                 ):
        super().__init__()

        # config
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = PositionalEncoding(config),
                h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )

        # init weights
        self.apply(self._init_weights)

        # output dim
        self.output_dim = config.n_embd

    # forward pass
    def forward(self, idx):

        # split idx into batch dim and sequence dim
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}; block size is {self.config.block_size}"

        x = self.transformer.wpe(self.transformer.wte(idx)) # add pos emb to token embedding input since it's static

        # forward through blocks
        for block in self.transformer.h:
            x = block(x)

        # forward through final layer norm
        x = self.transformer.ln_f(x)

        # output
        x = x.mean(dim=1) # mean pool over sequence length --> (B, n_embd)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
# ----------------------------------------------------------------
# create MLP encoder for environmental covariates
class E_Encoder(nn.Module):

    def __init__(self,
                 input_dim: int = 374,
                 output_dim: int = 128,
                 hidden_dim: int = 256,
                 n_hidden: int = 4,
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.1,
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

# ----------------------------------------------------------------
# create full GxE transformer for genomic prediction
class GxE_Transformer(nn.Module):

    """
    Full transformer for genomic prediction
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
            g_enc = self.g_encoder(x["g_data"])
            e_enc = self.e_encoder(x["e_data"])
            assert g_enc.shape == e_enc.shape, "G and E encoders must output same shape"
            x = g_enc + e_enc
        elif self.g_enc_flag:
            x = self.g_encoder(x["g_data"])
        else:
            x = self.e_encoder(x["e_data"])
        for layer in self.hidden_layers:
            x = x + layer(x)

        return self.final_layer(x)