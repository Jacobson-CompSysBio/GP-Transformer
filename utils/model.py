import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# look at Chinchilla for parameter recommendations
# ----------------------------------------------------------------
# create MLP block for flat layers
class Block(nn.Module):
    """
    Flat block with a single linear layer, batchnorm, dropout and activation
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.1,
                 batchnorm: bool = True,
                 ):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(output_dim)
    
    # fwd pass
    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        if hasattr(self, 'batchnorm'):
            x = self.batchnorm(x)
        return x

# ----------------------------------------------------------------
# config for transformer
@dataclass
class TransformerConfig:
    block_size: int = 2240 # max sequence length
    vocab_size: int = 4 # 0, 1, 2, 3
    n_layer: int = 2 # number of transformer layers
    n_head: int = 4 # number of heads
    n_embd: int = 128 # embedding size 

# transformer
class SelfAttention(nn.Module):
    """
    Bidirectional self attention with FlashAttention
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "hidden_dim must be divisible by head_size"

        # init k, q, v linear layers (all in same layer)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # get attributes
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # for this project, B = batch size, T = 2024 (sequence length), and C = 768 (hidden_dim)
        B, T, C = x.size()

        """
        calculate query, key, and value for all heads in batch and move head forward to be batched
        nh is "number of heads", hs is "head size", and C (# channels) = nh * hs
        e.g. GPT2: nh = 12, hs = 64, so nh * hs = C = 768 channels
        """

        # get q, k, v
        qkv = self.c_attn(x) # (B, T, C * 3)

        # split and transpose for faster computation
        # pytorch process B and num_heads dim as batches, processes in parallel
        q, k, v = qkv.split(self.n_embd, dim=2) # split along T dimension
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False) # no mask, so is_causal=False
        y = y.transpose(1, 2).contiguous().view(B, T, C) # reassemble head outputs side-by-side

        # output projection
        return self.c_proj(y)

class TransformerMLP(nn.Module):
    """
    MLP for transformer
    """
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = TransformerMLP(config)
    
    def forward(self, x):
        # resid connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# create transformer encoder for genotype data
class G_Encoder(nn.Module):

    def __init__(self, config
                 ):
        super().__init__()

        # config
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, 1, bias=False)

        self.f_proj = nn.Linear(config.block_size, config.n_embd, bias=False)

        # output dim
        self.output_dim = config.n_embd

        # weight sharing + weight init

    # forward pass
    def forward(self, idx):

        # split idx into batch dim and sequence dim
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}; block size is {self.config.block_size}"

        # embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape T
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        # forward through blocks
        for block in self.transformer.h:
            x = block(x)

        # forward through final layer norm
        x = self.transformer.ln_f(x)

        # output
        x = self.lm_head(x) # shape (B, T, 1)
        x = self.f_proj(x.squeeze(-1)) # shape (B, n_embd=768)

        return x
    
# ----------------------------------------------------------------
# create MLP encoder for environmental covariates
class E_Encoder(nn.Module):

    def __init__(self,
                 input_dim: int = 374,
                 output_dim: int = 128,
                 hidden_dim: int = 128,
                 n_hidden: int = 2,
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.1,
                 batchnorm: bool = True,
                 ):
        super().__init__()

        # set attributes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden
        self.activation = activation
        self.dropout = dropout
        self.batchnorm = batchnorm
        
        # make hidden layers
        self.hidden_layers = nn.ModuleList([Block(input_dim, hidden_dim, activation, dropout, batchnorm) if i == 0 else Block(hidden_dim, hidden_dim, activation, dropout, batchnorm) for i in range(n_hidden)])

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
                 hidden_dim: int = 128,
                 n_hidden: int =  1,
                 hidden_activation: nn.Module = nn.GELU(),
                 final_activation: nn.Module = nn.Identity(),
                 g_enc: bool = True,
                 e_enc: bool = True,
                 config = None
                 ):
        super().__init__()

        # set attributes
        self.g_enc = g_enc
        self.e_enc = e_enc
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden
        self.hidden_activation = hidden_activation
        self.dropout = dropout

        # init G, E encoders
        if self.g_enc:
            self.g_encoder = G_Encoder(config)
        if self.e_enc:
            self.e_encoder = E_Encoder()

        # get output dimensions of G, E encoders (should be the same)
        if g_enc:
            self.gxe_output_dim = self.g_encoder.output_dim
        else:
            self.gxe_output_dim = self.e_encoder.output_dim

        # make hidden layers
        self.hidden_layers = nn.ModuleList([Block(self.gxe_output_dim, hidden_dim, hidden_activation, dropout) if i == 0 else Block(hidden_dim, hidden_dim, hidden_activation, dropout) for i in range(n_hidden)])
        
        # init final layer (output of 1 for regression)
        self.final_layer = nn.Linear(hidden_dim, 1) # CAN CHANGE INPUT, OUTPUT SIZE FOR LAYERS
        self.final_layer_activation = final_activation

    def forward(self, x):

        # only pass through G, E encoders if they exist
        if hasattr(self, 'g_encoder') and hasattr(self, 'g_encoder'):

            # separate x vals
            g = x["g_data"]
            e = x["e_data"]

            # pass through G, E encoders
            g_enc = self.g_encoder(g)
            e_enc = self.e_encoder(e)

            # concatenate encodings
            x = g_enc + e_enc

        elif hasattr(self, 'g_encoder'):

            # separate x vals
            g = x["g_data"]

            # pass through G encoder
            x = self.g_encoder(g)

        elif hasattr(self, 'e_encoder'):
            # separate x vals
            e = x["e_data"]

            # pass through E encoder
            x = self.e_encoder(e)

        # pass through other layers
        for layer in self.hidden_layers:
            x = x + layer(x)

        # pass through final layer + activation
        x = self.final_layer_activation(self.final_layer(x))

        return x