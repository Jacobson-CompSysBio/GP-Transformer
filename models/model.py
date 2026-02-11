import os, sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.cnn import *
from models.mlp import *
from models.transformer import * 

# ----------------------------------------------------------------
# Full transformer that treats markers + env covariates as tokens
class FullTransformer(nn.Module):
    def __init__(self,
                 config,
                 mlp_type: str = "dense",
                 moe_num_experts: int = 4,
                 moe_top_k: int = 2,
                 moe_expert_hidden_dim: Optional[int] = None,
                 moe_shared_expert: bool = False,
                 moe_shared_expert_hidden_dim: Optional[int] = None,
                 moe_loss_weight: float = 0.01):
        super().__init__()
        self.config = config
        if isinstance(mlp_type, str):
            self.mlp_type = mlp_type.lower()
        else:
            self.mlp_type = "moe" if mlp_type else "dense"
        if self.mlp_type not in {"dense", "moe"}:
            raise ValueError(f"mlp_type must be 'dense' or 'moe' (got {mlp_type})")
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_expert_hidden_dim = moe_expert_hidden_dim
        self.moe_shared_expert = moe_shared_expert
        self.moe_shared_expert_hidden_dim = moe_shared_expert_hidden_dim
        self.moe_loss_weight = moe_loss_weight
        self.moe_aux_loss = None
        self.g_input_type = str(getattr(config, "g_input_type", "tokens")).lower()
        if self.g_input_type not in {"tokens", "grm"}:
            raise ValueError(f"config.g_input_type must be 'tokens' or 'grm' (got {self.g_input_type})")

        # tokenizers
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        if self.g_input_type == "tokens":
            self.g_embed = nn.Embedding(config.vocab_size, config.n_embd)
            self.g_proj = None
        else:
            self.g_embed = None
            self.g_proj = nn.Linear(1, config.n_embd)
        # project each scalar env feature to a token embedding
        self.e_proj = nn.Linear(1, config.n_embd)

        # positional encoding for combined marker + env + cls tokens
        max_len = config.block_size + config.n_env_fts + 1
        class _PE(nn.Module):
            def __init__(self, n_embd, max_len, dropout):
                super().__init__()
                pe = torch.zeros(max_len, n_embd)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10_000.0) / n_embd))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe, persistent=False)
                self.dropout = nn.Dropout(dropout)
            def forward(self, x):
                x = x + self.pe[:, :x.shape[1], :].to(x.dtype)
                return self.dropout(x)
        self.wpe = _PE(config.n_embd, max_len, config.dropout)

        # transformer body
        dpr = [x.item() for x in torch.linspace(0, config.dropout * 0.5, config.n_gxe_layer)]
        if self.mlp_type == "moe":
            def build_block(i):
                return TransformerMoEBlock(
                    config,
                    num_experts=self.moe_num_experts,
                    k=self.moe_top_k,
                    expert_hidden_dim=self.moe_expert_hidden_dim,
                    shared_expert=self.moe_shared_expert,
                    shared_expert_hidden_dim=self.moe_shared_expert_hidden_dim,
                    drop_path=dpr[i],
                )
        else:
            def build_block(i):
                return TransformerBlock(config, drop_path=dpr[i])
        self.blocks = nn.ModuleList([build_block(i) for i in range(config.n_gxe_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        
        # Projection head for contrastive learning (like SimCLR)
        # Projects G embeddings to a space good for contrastive learning
        self.g_proj = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd // 2),
        )

    def _build_tokens(self, x):
        g = x["g_data"]            # (B, Tm)
        e = x["e_data"]            # (B, Feats)
        B, Tm = g.shape
        # embeddings
        if self.g_input_type == "tokens":
            g_tok = self.g_embed(g.long())             # (B, Tm, C)
        else:
            g_tok = self.g_proj(g.float().unsqueeze(-1))  # (B, Tm, C)
        e_tok = self.e_proj(e.unsqueeze(-1))           # (B, Feats, C)
        cls = (self.cls_token + self.cls_pos).expand(B, -1, -1)
        tokens = torch.cat([cls, g_tok, e_tok], dim=1) # (B, 1+Tm+Feats, C)
        tokens = self.wpe(tokens)
        env_start = 1 + Tm
        return tokens, env_start

    def _encode_tokens(self, tokens):
        self.moe_aux_loss = None
        if self.mlp_type == "moe":
            moe_loss_sum = None
            moe_loss_count = 0
            if self.moe_loss_weight and self.moe_loss_weight != 0:
                from utils.loss import moe_load_balance_loss
            for blk in self.blocks:
                tokens, gate_weights = blk(tokens)
                if gate_weights is not None and self.moe_loss_weight and self.moe_loss_weight != 0:
                    loss = moe_load_balance_loss(gate_weights, self.moe_num_experts)
                    moe_loss_sum = loss if moe_loss_sum is None else (moe_loss_sum + loss)
                    moe_loss_count += 1
            if moe_loss_count > 0 and self.moe_loss_weight and self.moe_loss_weight != 0:
                self.moe_aux_loss = (moe_loss_sum / moe_loss_count) * self.moe_loss_weight
        else:
            for blk in self.blocks:
                tokens = blk(tokens)
        tokens = self.ln_f(tokens)
        return tokens

    def forward(self, x, return_g_embeddings: bool = False):
        tokens, env_start = self._build_tokens(x)
        tokens = self._encode_tokens(tokens)
        
        # Extract G embeddings AFTER transformer (for contrastive loss)
        # This is crucial - now the embeddings contain learned representations
        if return_g_embeddings:
            # G tokens are from index 1 to env_start (excluding CLS at 0)
            g_tokens = tokens[:, 1:env_start, :]  # (B, Tm, C)
            g_pooled = g_tokens.mean(dim=1)  # Pool to (B, C)
            # Project through contrastive head (like SimCLR)
            g_embed = self.g_proj(g_pooled)  # (B, C//2)
        
        pred = self.head(tokens[:, 0])
        
        if return_g_embeddings:
            return pred, g_embed
        return pred

    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")

# full transformer with residual heads (env mean + residual = total)
class FullTransformerResidual(FullTransformer):

    """
    """

    def __init__(self,
                 config,
                 mlp_type: str = "dense",
                 moe_num_experts: int = 4,
                 moe_top_k: int = 2,
                 moe_expert_hidden_dim: Optional[int] = None,
                 moe_shared_expert: bool = False,
                 moe_shared_expert_hidden_dim: Optional[int] = None,
                 moe_loss_weight: float = 0.01,
                 residual: bool = False):
        super().__init__(
            config=config,
            mlp_type=mlp_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_expert_hidden_dim=moe_expert_hidden_dim,
            moe_shared_expert=moe_shared_expert,
            moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
            moe_loss_weight=moe_loss_weight,
        )
        self.residual = residual
        self.detach_ymean_in_sum = False
        self.ymean_head = nn.Linear(config.n_embd, 1)
        
        # Projection head for contrastive learning (like SimCLR)
        self.g_proj = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd // 2),
        )

    def forward(self, x, return_g_embeddings: bool = False):
        tokens, env_start = self._build_tokens(x)
        
        # IMPORTANT: Encode FIRST, then extract features from encoded tokens
        tokens = self._encode_tokens(tokens)
        resid_pred = self.head(tokens[:, 0])
        
        # Compute ymean from ENCODED env tokens (not raw projections!)
        ymean_pred = None
        if env_start < tokens.size(1):
            env_tokens = tokens[:, env_start:, :]  # Now these are encoded
            env_repr = env_tokens.mean(dim=1)
            ymean_pred = self.ymean_head(env_repr)

        if ymean_pred is None:
            ymean_pred = torch.zeros_like(resid_pred)
        
        # Extract G embeddings for contrastive loss
        g_embed = None
        if return_g_embeddings:
            g_tokens = tokens[:, 1:env_start, :]  # G tokens (excluding CLS)
            g_pooled = g_tokens.mean(dim=1)  # Pool to (B, C)
            g_embed = self.g_proj(g_pooled)  # (B, C//2)

        if not self.residual:
            if return_g_embeddings:
                return resid_pred, g_embed
            return resid_pred

        if self.detach_ymean_in_sum and isinstance(ymean_pred, torch.Tensor):
            total_pred = ymean_pred.detach() + resid_pred
        else:
            total_pred = ymean_pred + resid_pred
        result = {'total': total_pred,
                'ymean': ymean_pred,
                'resid': resid_pred,
        }
        if return_g_embeddings:
            return result, g_embed
        return result

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
                 g_encoder_type: str = None,
                 moe_num_experts: int = None,
                 moe_top_k: int = None,
                 moe_expert_hidden_dim: int = None,
                 moe_shared_expert: bool = None,
                 moe_shared_expert_hidden_dim: int = None,
                 moe_loss_weight: float = None,
                 config = None
                 ):
        super().__init__()

        self.config = config
        self.g_encoder_type = g_encoder_type if g_encoder_type is not None else getattr(config, "g_encoder_type", "dense")
        if isinstance(self.g_encoder_type, str):
            self.g_encoder_type = self.g_encoder_type.lower()
        else:
            self.g_encoder_type = "moe" if self.g_encoder_type else "dense"
        self.moe_num_experts = moe_num_experts if moe_num_experts is not None else getattr(config, "moe_num_experts", 4)
        self.moe_top_k = moe_top_k if moe_top_k is not None else getattr(config, "moe_top_k", 2)
        self.moe_expert_hidden_dim = moe_expert_hidden_dim if moe_expert_hidden_dim is not None else getattr(config, "moe_expert_hidden_dim", None)
        self.moe_shared_expert = moe_shared_expert if moe_shared_expert is not None else getattr(config, "moe_shared_expert", False)
        self.moe_shared_expert_hidden_dim = moe_shared_expert_hidden_dim if moe_shared_expert_hidden_dim is not None else getattr(config, "moe_shared_expert_hidden_dim", None)
        self.moe_loss_weight = moe_loss_weight if moe_loss_weight is not None else getattr(config, "moe_loss_weight", 0.01)
        self.g_input_type = str(getattr(config, "g_input_type", "tokens")).lower()
        self.use_moe_encoder = bool(g_enc) and self.g_encoder_type == "moe"
        self.moe_aux_loss = None

        # set attributes
        self.g_encoder = (
            G_Encoder(
                config,
                encoder_type=self.g_encoder_type,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_expert_hidden_dim=self.moe_expert_hidden_dim,
                moe_shared_expert=self.moe_shared_expert,
                moe_shared_expert_hidden_dim=self.moe_shared_expert_hidden_dim,
            )
            if g_enc else None
        )
        self.e_encoder = E_Encoder(input_dim=config.n_env_fts,
                                   output_dim=config.n_embd,
                                   hidden_dim=config.n_embd,
                                   n_hidden=config.n_mlp_layer,
                                   dropout=config.dropout) if e_enc else None
        self.ld_encoder = LD_Encoder(input_dim=config.vocab_size,
                                     output_dim=config.n_embd,
                                     num_blocks=config.n_ld_layer,
                                     dropout=config.dropout) if ld_enc else None

        self.moe_w = nn.Parameter(torch.zeros(3)) if moe else None # logits
        self.fuse_ln = nn.LayerNorm(config.n_embd) # add ln for moe fusion

        # append env as a token to final tf layer instead of adding to all reprs
        self.env_as_token = True  # set to false for old behavior
        self.detach_ymean_in_sum = False  # whether to detach ymean prediction in residual sum
        
        # stochastic depth: linearly increase drop rate from 0 to max
        drop_path_rate = config.dropout * 0.5  # max drop path rate
        
        if gxe_enc == "tf":
            self.gxe_enc = "tf"
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, config.n_gxe_layer)]
            self.hidden_layers = nn.ModuleList(
            [TransformerBlock(config, drop_path=dpr[i]) for i in range(config.n_gxe_layer)]
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

        # init final layer with dropout for regularization
        self.final_dropout = nn.Dropout(config.dropout)
        self.final_layer = nn.Linear(config.n_embd, 1)
        
        # initialize final layer with small weights for stable training
        nn.init.normal_(self.final_layer.weight, std=0.01)
        nn.init.zeros_(self.final_layer.bias)

    # concat now uses softmax to augment moe weights proportionally 
    def _concat(self, g_enc, e_enc, ld_enc):
        w = torch.softmax(self.moe_w, dim=0) if self.moe_w is not None else None
        x = 0
        if isinstance(g_enc, torch.Tensor):
            x = x + (w[0] * g_enc if w is not None else g_enc)
        if isinstance(e_enc, torch.Tensor):
            x = x + (w[1] * e_enc if w is not None else e_enc)
        if isinstance(ld_enc, torch.Tensor):
            x = x + (w[2] * ld_enc if w is not None else ld_enc)
        return self.fuse_ln(x)

    def _forward_tf(self, g_enc, e_enc, ld_enc):
        if isinstance(e_enc, torch.Tensor):
            if self.env_as_token:
                # append single env token
                e_tok = e_enc.unsqueeze(dim=1)  # [B, 1, C]
                x = torch.cat([g_enc, e_tok], dim=1) if isinstance(g_enc, torch.Tensor) else e_tok

                # align ld_enc by padding to match x's sequence length
                if isinstance(ld_enc, torch.Tensor):
                    seq_diff = x.size(1) - ld_enc.size(1)
                    if seq_diff > 0:
                        pad = torch.zeros(ld_enc.size(0), seq_diff, ld_enc.size(2), device=ld_enc.device, dtype=ld_enc.dtype)
                        ld_enc = torch.cat([ld_enc, pad], dim=1)
                    x = self.fuse_ln(x + ld_enc)
            else:
                e_map = e_enc.unsqueeze(dim=1).expand(-1, g_enc.size(1), -1)
                x = self._concat(g_enc, e_map, ld_enc)
        else:
            x = self._concat(g_enc, e_enc, ld_enc)
        for layer in self.hidden_layers:
            x = layer(x)
        # take CLS (first token) as summary if present; else mean
        if isinstance(x, torch.Tensor) and x.size(1) > 0:
            return x[:, 0]  # (B, C)
        return x.mean(dim=1)

    def _forward_mlp(self, g_enc, e_enc, ld_enc):
        # convert [B, T, C] -> [B, C]
        if isinstance(g_enc, torch.Tensor):
            g_main = g_enc[:, 1:] if g_enc.size(1) > 1 else g_enc
            g_enc = g_main.mean(dim=1)
        if isinstance(ld_enc, torch.Tensor):
            ld_main = ld_enc[:, 1:] if ld_enc.size(1) > 1 else ld_enc
            ld_enc = ld_main.mean(dim=1)
        x = self._concat(g_enc, e_enc, ld_enc)
        for layer in self.hidden_layers:
            x = x + layer(x)
        return x
    
    def _forward_cnn(self, g_enc, e_enc, ld_enc):
        #unsqueeze e to concat properly
        if isinstance(e_enc, torch.Tensor):
            e_enc = e_enc.unsqueeze(dim=1)
            if isinstance(g_enc, torch.Tensor):
                e_enc = e_enc.expand(-1, g_enc.size(1), -1)
        x = self._concat(g_enc, e_enc, ld_enc)

        x = x.transpose(1, 2) # (B, T, C) -> (B, C, T)
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.transpose(1, 2) # (B, C, T) -> (B, T, C)
        # drop CLS for pooling if present
        x = x[:, 1:] if x.size(1) > 1 else x
        x = x.mean(dim=1) # (B, T, n_embd) -> (B, n_embd) 
        return x

    def _encode(self, x):
        self.moe_aux_loss = None
        if self.g_encoder:
            if self.use_moe_encoder:
                g_enc, moe_loss = self.g_encoder(x["g_data"], return_moe_loss=True)
                if moe_loss is not None:
                    self.moe_aux_loss = moe_loss * self.moe_loss_weight
            else:
                g_enc = self.g_encoder(x["g_data"])
        else:
            g_enc = 0
        e_enc = self.e_encoder(x["e_data"]) if self.e_encoder else 0
        ld_enc = 0
        if self.ld_encoder:
            if self.g_input_type != "tokens":
                raise ValueError("LD encoder requires tokenized genotype inputs (g_input_type='tokens').")
            ld_feats = F.one_hot(x["g_data"].long(), num_classes=self.ld_encoder.input_dim)
            ld_enc = self.ld_encoder(ld_feats.float())
            # pad CLS for ld_enc to match g_enc length if needed
            if isinstance(g_enc, torch.Tensor) and ld_enc.dim() == 3 and ld_enc.size(1) + 1 == g_enc.size(1):
                pad = torch.zeros(ld_enc.size(0), 1, ld_enc.size(2), device=ld_enc.device, dtype=ld_enc.dtype)
                ld_enc = torch.cat([pad, ld_enc], dim=1)

        if self.gxe_enc == "tf":
            rep = self._forward_tf(g_enc, e_enc, ld_enc)
        elif self.gxe_enc == "mlp":
            rep = self._forward_mlp(g_enc, e_enc, ld_enc)
        elif self.gxe_enc == "cnn":
            rep = self._forward_cnn(g_enc, e_enc, ld_enc)
        else:
            raise ValueError("gxe_enc must be one of ['tf', 'mlp', 'cnn']")

        return rep, e_enc, g_enc
    
    def forward(self, x, return_g_embeddings: bool = False):
        rep, _, g_enc = self._encode(x)
        pred = self.final_layer(self.final_dropout(rep))
        
        if return_g_embeddings:
            # Return pooled G embedding for contrastive loss
            if isinstance(g_enc, torch.Tensor):
                if g_enc.dim() == 3:
                    # Take CLS token or mean pool
                    g_embed = g_enc[:, 0] if g_enc.size(1) > 1 else g_enc.mean(dim=1)
                else:
                    g_embed = g_enc
            else:
                g_embed = None
            return pred, g_embed
        
        return pred
    
    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        g_trainable_params = sum(p.numel() for p in self.g_encoder.parameters() if p.requires_grad) if self.g_encoder else 0
        e_trainable_params = sum(p.numel() for p in self.e_encoder.parameters() if p.requires_grad) if self.e_encoder else 0
        ld_trainable_params = sum(p.numel() for p in self.ld_encoder.parameters() if p.requires_grad) if self.ld_encoder else 0
        gxe_trainable_params = trainable_params - g_trainable_params - e_trainable_params - ld_trainable_params
        print(f"Trainable parameters: {trainable_params:,}"
              f" (G: {g_trainable_params:,}, E: {e_trainable_params:,}, LD: {ld_trainable_params:,}, GxE: {gxe_trainable_params:,})")


# create full GxE transformer for genomic prediction
class GxE_ResidualTransformer(GxE_Transformer):

    """
    """

    def __init__(self,
                 g_enc: bool = True,
                 e_enc: bool = True,
                 ld_enc: bool = True,
                 gxe_enc: str = "tf",
                 moe: bool = True,
                 residual: bool = False,
                 g_encoder_type: str = None,
                 moe_num_experts: int = None,
                 moe_top_k: int = None,
                 moe_expert_hidden_dim: int = None,
                 moe_shared_expert: bool = None,
                 moe_shared_expert_hidden_dim: int = None,
                 moe_loss_weight: float = None,
                 config = None
                 ):
        super().__init__(
            g_enc=g_enc,
            e_enc=e_enc,
            ld_enc=ld_enc,
            gxe_enc=gxe_enc,
            moe=moe,
            g_encoder_type=g_encoder_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_expert_hidden_dim=moe_expert_hidden_dim,
            moe_shared_expert=moe_shared_expert,
            moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
            moe_loss_weight=moe_loss_weight,
            config=config,
        )
        self.residual = residual
        # env head to predict per-year mean yield (if residual)
        self.ymean_head = nn.Linear(config.n_embd, 1) if self.e_encoder is not None else None
        
        # Projection head for contrastive learning (like SimCLR)
        self.g_proj = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd // 2),
        )

    def forward(self, x, return_g_embeddings: bool = False):
        rep, e_enc, g_enc = self._encode(x)  # Unpack all 3 return values

        # Extract G embeddings for contrastive loss
        g_embed = None
        if return_g_embeddings:
            if isinstance(g_enc, torch.Tensor):
                if g_enc.dim() == 3:
                    g_pooled = g_enc[:, 0] if g_enc.size(1) > 1 else g_enc.mean(dim=1)
                else:
                    g_pooled = g_enc
                g_embed = self.g_proj(g_pooled)  # (B, C//2)

        # non-residual mode: final layer predicts total yield
        if not self.residual:
            pred = self.final_layer(self.final_dropout(rep))
            if return_g_embeddings:
                return pred, g_embed
            return pred

        # residual mode:
        ymean_pred = self.ymean_head(e_enc) if self.ymean_head is not None else 0
        resid_pred = self.final_layer(self.final_dropout(rep))

        # option: detach ymean_pred to prevent env head from learning residual signal
        if self.detach_ymean_in_sum and isinstance(ymean_pred, torch.Tensor):
            total_pred = ymean_pred.detach() + resid_pred
        else:
            total_pred = ymean_pred + resid_pred
        result = {'total': total_pred,
                'ymean': ymean_pred,
                'resid': resid_pred,
        }
        if return_g_embeddings:
            return result, g_embed
        return result

# ----------------------------------------------------------------

# create full transformer with mixture of experts
class FullTransformerMoE(nn.Module):
    """
    Full Transformer with Mixture of Experts architecture.
    Uses attention mechanism across all variables, G and E.
    CLS token used for final yield prediction.
    """
    def __init__(self, config):
        super().__init__()
    
    def forward(self, x):
        pass

    def print_trainable_parameters(self):
        pass
