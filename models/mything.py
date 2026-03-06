import torch
import torch.nn as nn
import torch.nn.functional as nnF


class GeneralAttention(nn.Module):
    def __init__(self, config, windowed=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "hidden_dim must be divisible by head_size"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.windowed = windowed
        self.window_size = 64

        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    
    def forward(self):
        raise NotImplementedError
    
    def _build_window_mask(self, T, device):
        idx = torch.arange(T, device=device)

        dist = idx[None, :] - idx[:, None]
        mask = dist.abs() > self.window_size

        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask
    
    def _reshape(self, B, q, k, v):
        return (
            q.view(B, -1, self.n_head, self.head).transpose(1, 2), # (B, num_heads, T, head_size)
            k.view(B, -1, self.n_head, self.head).transpose(1, 2), # (B, num_heads, T, head_size)
            v.view(B, -1, self.n_head, self.head).transpose(1, 2) # (B, num_heads, T, head_size)
        )
    
    def _flash_attn(self, B, T, q, k, v):
        # flash attention with dropout
        dropout_p = self.dropout if self.training else 0.0
        y = nnF.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
            dropout_p=dropout_p,
            attn_mask=self._build_window_mask(T, q.device) if self.windowed else None
        )
        return y.transpose(1, 2).contiguous().view(B, -1, self.n_embd) # reassemble head outputs side-by-side    


class SelfAttention(GeneralAttention):
    def __init__(self, config, windowed=False):
        super().__init__(config, windowed=windowed)

        # init k, q, v linear layers (all in same layer)
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)

    def forward(self, x):
        B, T, _ = x.shape

        # get q, k, v
        qkv = self.c_attn(x) # (B, T, C * 3)

        # split and transpose for faster computation
        # pytorch process B and num_heads dim as batches, processes in parallel
        q, k, v = qkv.chunk(3, dim=2) # split along channels 
        q, k, v = self._reshape(B, q, k, v)

        y = self._flash_attn(B, T, q, k, v)

        # output projection
        return self.c_proj(y)


class CrossAttention(GeneralAttention):
    def __init__(self, config):
        super().__init__(config)

        # q, k, v projections
        self.W_q = nn.Linear(config.n_embd, config.n_embd)
        self.W_kv = nn.Linear(config.n_embd, config.n_embd * 2)

    def forward(self, q, k):
        B = q.size(0)

        # get q, k, v projections
        q = self.W_q(q)
        kv = self.W_kv(k) #.unsqueeze(1)

        k, v = kv.chunk(2, dim=-1)
        q, k, v = self._reshape(B, q, k, v)

        # flash attention with dropout
        y = self._flash_attn(B, 0, q, k, v) # not windowed, so T doesn't matter

        # output projection
        return self.c_proj(y)


class EnvConditioning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gamma = nn.Linear(config.n_embd, config.n_embd)
        self.beta = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, snp_tokens, env_emb):
        gamma = self.gamma(env_emb).unsqueeze(1)
        beta = self.beta(env_emb).unsqueeze(1)

        return gamma * snp_tokens + beta


class GTF(nn.Module):
    def __init__(self, config, windowed_attn=False):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.film = EnvConditioning(config)
        self.attn = SelfAttention(config, windowed=windowed_attn)
        
        self.ln2 = nn.LayerNorm(config.n_embd)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, env):
        attn_input = self.ln1(x)
        attn_input_cond = self.film(attn_input, env)
        attn_output = self.attn(attn_input_cond)
        x = x + self.dropout(attn_output)

        ffn_input = self.ln2(x)
        ffn_output = self.ffn(ffn_input)
        x = x + self.dropout(ffn_output)

        return x


class EMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(config.n_env_fts, 512)
        self.fc2 = nn.Linear(512, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = nnF.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class MyThing(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.snp_embd = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embd = nn.Embedding(config.block_size, config.n_embd)

        self.e_enc = EMLP(config)
        self.g_ld = GTF(config, windowed_attn=True)
        self.g_epi = GTF(config, windowed_attn=False)
        
        self.eln = nn.LayerNorm(config.n_embd)
        self.gln = nn.LayerNorm(config.n_embd)

        self.crossattn = CrossAttention(config)

        self.yield_pred_head = nn.Sequential(
            nn.Linear(config.n_embd, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        g = x["g_data"]
        e = x["e_data"]
        B, g_T = g.shape
        _, e_T = e.shape

        # E covariate processing (B, e_T -> B, n_embd)
        e = self.e_enc(e) # embedding
        e = self.eln(e) # norm

        # SNP processing (B, g_T -> B, g_T, n_embd)
        pos = torch.arange(g_T, device=g.device)
        g = self.snp_embd(g) + self.pos_embd(pos) # embedding
        g = self.g_ld(g, e) # ld tf block
        g = self.g_epi(g, e) # epistasis tf block
        g = self.gln(g) # norm

        # G tokens cross-attend to E vector (B, g_T, n_embd -> no change)
        gxe = self.crossattn(g, e) # q, k

        # Aggregate tokens (B, g_T, n_embd -> B, 1, n_embd)
        gxe = gxe.mean(dim=1) # currently using mean pooling

        # Make yield prediction (B, 1, n_embd -> B, 1)
        return self.yield_pred_head(gxe)

    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
