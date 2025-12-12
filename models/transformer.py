import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# ----------------------------------------------------------------
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
        self.dropout = nn.Dropout(p=config.dropout)
    
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
        self.dropout = config.dropout
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

        # flash attention with dropout
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=dropout_p)
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
    
    def __init__(self, config, drop_path: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = TransformerMLP(config)
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = drop_path  # stochastic depth rate

    def forward(self, x):
        # stochastic depth: randomly skip this block during training
        if self.training and self.drop_path > 0.0:
            if torch.rand(1).item() < self.drop_path:
                return x  # skip entire block
        
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

        # learned CLS embedding (summary token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        # optional positional bias for CLS (keeps length alignment without resizing PE)
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, config.n_embd))

        # stochastic depth: linearly increase drop rate
        drop_path_rate = config.dropout * 0.5
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, config.n_g_layer)]
        
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = PositionalEncoding(config),
                h = nn.ModuleList([TransformerBlock(config, drop_path=dpr[i]) for i in range(config.n_g_layer)]),
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

        # token + positional embeddings
        x = self.transformer.wpe(self.transformer.wte(idx)) # (B, T, C)

        # prepend CLS token (add small positional bias so it isn't identical to data tokens)
        cls = (self.cls_token + self.cls_pos).expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, C)

        # forward through blocks
        for block in self.transformer.h:
            x = block(x)

        # forward through final layer norm
        x = self.transformer.ln_f(x)

        # output
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
