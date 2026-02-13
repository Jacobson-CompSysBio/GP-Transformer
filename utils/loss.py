import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torchsort import soft_rank
from typing import Callable, Union, Tuple, Dict, List, Optional

# small helper for optional DDP all-reduce
def _all_reduce_if_needed(*tensors, op=dist.ReduceOp.SUM):
    if not (dist.is_available() and dist.is_initialized()):
        return tensors
    reduced = []
    for t in tensors:
        buf = t.clone()
        dist.all_reduce(buf, op=op)
        reduced.append(buf)
    return tuple(reduced)

### helpers ###
def soft_rank_1d(x: torch.Tensor, rs: float = 1.0):
    """
    1d wrapper for torchsort.soft_rank
    """
    x2 = x.unsqueeze(0)
    r2 = soft_rank(x2, regularization_strength=rs)
    return r2.squeeze(0)

### load-balancing loss for MoE ###
def moe_load_balance_loss(gate_weights: torch.Tensor, num_experts: int):
    """
    Calculates load balancing loss; makes sure tokens are evenly distributed across experts.

    Args:
        gate_weights: Tensor of shape [batch_size * seq_len, num_experts]
        num_experts: Total number of experts
    
    Returns:
        scalar loss tensor
    """

    # gate weights is the output of the gate network (before top-k)
    # need average routing probability per expert
    # and fraction of tokens routed to each expert

    num_tokens = gate_weights.shape[0]

    # calculate fraction of tokens routed to each expert (f_i)
    # use weights as a proxy for assignment count
    # (sum of weights for each expert across all tokens)
    tokens_per_expert = gate_weights.sum(dim=0) # shape: [num_experts]
    f_i = tokens_per_expert / num_tokens 

    # calculate average routing probability per expert (P_i)
    # this is mean of the gate weights for each expert
    P_i = torch.mean(gate_weights, dim=0) # shape: num_experts

    # calculate the loss: alpha * num_experts * sum(f_i * P_i)
    loss = num_experts * torch.sum(f_i * P_i)
    return loss

### per-env losses ###
def envwise_spearman(pred, target, env_id):
    """
    pred: [B, 1] or [B] (float)
    target: [B, 1] or [B] (float)
    env_id: [B] (long) 

    returns a scalar loss (mean(spearman over envs in batch))
    """
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if target.ndim > 1:
        target = target.squeeze(-1)
    
    device = pred.device
    env_id = env_id.to(device) 

    # counters for accum loss, count 
    spearmans = []

    # loop through env
    for env in torch.unique(env_id):
        mask = (env_id == env)
        
        # if < 2 obs, don't use 
        if int(mask.sum()) < 2:
            continue

        px = pred[mask]
        tx = target[mask]

        # if variances are 0, skip env
        if px.var(unbiased=False) == 0 or tx.var(unbiased=False) == 0:
            continue
       
        px_rank = soft_rank_1d(px)
        tx_rank = soft_rank_1d(tx)
        n = px_rank.numel()

        d2_sum = (px_rank - tx_rank).pow(2).sum()
        denom = torch.tensor(n * (n ** 2 - 1), dtype=torch.float32, device=device)

        rho = 1.0 - ((6.0 * d2_sum) / denom)
        spearmans.append(rho)

    # no valid groups in batch; return 1.0 loss contribution 
    if len(spearmans) == 0:
        # return a tensor that maintains grad flow - multiply pred by 0 and add 1.0
        # This ensures gradients can flow back through pred while returning loss=1.0
        return (pred.sum() * 0.0) + 1.0

    # get 1.0 - avg spearman as loss 
    return 1.0 - torch.stack(spearmans).mean() 

def envwise_mse(pred, target, env_id, eps: float = 1e-8, min_samples: int = 2):
    """
    Macro-averaged per-environment MSE.
    Computes MSE within each environment, then averages equally across environments.
    This gives equal weight to each environment regardless of sample count,
    matching the macro-avg methodology used in eval.py for test metrics.
    """
    # squeeze preds, targets if they are [B, 1]
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if target.ndim > 1:
        target = target.squeeze(-1)

    device = pred.device
    env_id = env_id.to(device)
    pred_f = pred.float()
    target_f = target.float()

    max_env = int(env_id.max().item()) + 1

    def _accumulate(vec: torch.Tensor) -> torch.Tensor:
        buf = vec.new_zeros(max_env)
        return buf.scatter_add(0, env_id, vec)

    # Per-sample squared error
    se = (pred_f - target_f) ** 2

    # Accumulate count and sum-of-SE per environment
    count = _accumulate(torch.ones_like(pred_f))
    sum_se = _accumulate(se)

    # Only include environments with enough samples
    valid = count >= min_samples
    if not valid.any():
        # fallback to global MSE
        return se.mean()

    # Per-env MSE = sum_se / count, then macro-average across valid envs
    per_env_mse = sum_se[valid] / count[valid]
    return per_env_mse.mean()

def envwise_pcc(pred, target, env_id, eps=1e-8, min_samples=4):
    """
    Compute Pearson r independently for each environment.
    Computes LOCAL correlation per environment - DDP handles gradient sync.
    Uses Fisher z-transform weighted by sample count for stable averaging.
    
    IMPORTANT: For this loss to work well during training, batches should
    contain multiple samples from the same environment. Use EnvStratifiedSampler
    instead of random sampling to ensure reliable gradients.
    
    Args:
        pred: Predictions [B] or [B, 1]
        target: Targets [B] or [B, 1]  
        env_id: Environment IDs [B]
        eps: Small constant for numerical stability
        min_samples: Minimum samples per environment to include in loss
                     (environments with fewer samples are excluded, default=4)
    """
    pred = pred.squeeze(-1).float()
    target = target.squeeze(-1).float()
    device = pred.device
    env_id = env_id.to(device)

    max_env = int(env_id.max().item()) + 1

    def _accumulate(vec: torch.Tensor) -> torch.Tensor:
        buf = vec.new_zeros(max_env)
        return buf.scatter_add(0, env_id, vec)

    # Compute LOCAL sufficient statistics only (no all-reduce to preserve gradients)
    ones = torch.ones_like(pred)
    count = _accumulate(ones)
    sx = _accumulate(pred)
    sy = _accumulate(target)
    sxx = _accumulate(pred ** 2)
    syy = _accumulate(target ** 2)
    sxy = _accumulate(pred * target)

    # Per-env correlation using local batch
    n = count.clamp_min(1.0)
    mean_x = sx / n
    mean_y = sy / n
    
    # Covariance and variances
    cov = sxy / n - mean_x * mean_y
    var_x = (sxx / n - mean_x ** 2).clamp_min(eps)
    var_y = (syy / n - mean_y ** 2).clamp_min(eps)
    
    # Pearson r per environment
    r_per_env = cov / (var_x.sqrt() * var_y.sqrt() + eps)
    r_per_env = r_per_env.clamp(-1.0, 1.0)

    # Require minimum sample count for stable correlation estimate
    valid = (count >= min_samples) & (var_x > eps) & (var_y > eps)
    if not valid.any():
        # fallback: use global Pearson correlation on local batch
        r = torch_pearsonr(pred, target)
        if not torch.isfinite(r).all():
            return (pred.sum() * 0.0) + 1.0
        return 1.0 - r

    # Uniform (macro) average across environments — matches eval.py's test metric
    r_valid = r_per_env[valid]
    r_bar = r_valid.mean()
    
    return 1.0 - r_bar


### other losses ### 
def torch_spearmanr(pred: torch.Tensor,
                    target: torch.Tensor,
                    dim: int = -1,
                    eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable Spearman: Pearson correlation of soft ranks along `dim`.
    Returns rho with the same shape as pred/target with `dim` removed (i.e., reduced).
    """
    # move the ranking dimension to the end
    if dim != -1:
        pred = pred.transpose(dim, -1)
        target = target.transpose(dim, -1)

    n = pred.size(-1)
    if n < 2:
        # not enough elements to define a rank correlation
        rho = torch.zeros(pred.shape[:-1], device=pred.device, dtype=pred.dtype)
        return rho if dim == -1 else rho.transpose(dim, -1)

    r_pred = soft_rank(pred, regularization_strength=1.0)
    r_targ = soft_rank(target, regularization_strength=1.0)

    # center ranks
    r_pred = r_pred - r_pred.mean(dim=-1, keepdim=True)
    r_targ = r_targ - r_targ.mean(dim=-1, keepdim=True)

    # norms; clamp to avoid division by zero when either side is constant
    denom = (r_pred.norm(dim=-1) * r_targ.norm(dim=-1)).clamp_min(eps)
    rho = (r_pred * r_targ).sum(dim=-1) / denom  # shape: pred.shape[:-1]

    return rho if dim == -1 else rho.transpose(dim, -1)

def torch_pearsonr(pred: torch.Tensor, target: torch.Tensor, dim=0, eps=1e-8):
    """
    pred, target: [N] or [N, D]; set dim=0 for batch-wise correlation.
    Returns r with shape [] if vectors, or [D] if multi-dim features.
    Non-DDP helper (local only).
    """
    pred = pred.float()
    target = target.float()

    # center
    pred = pred - pred.mean(dim=dim, keepdim=True)
    target = target - target.mean(dim=dim, keepdim=True)

    # sums of squares and covariance
    v_pred = (pred * pred).sum(dim=dim)
    v_target = (target * target).sum(dim=dim)
    cov = (pred * target).sum(dim=dim)

    r = cov / (v_pred.clamp_min(eps).sqrt() * v_target.clamp_min(eps).sqrt())
    return r.clamp(-1.0, 1.0)

def macro_env_pearson(
    pred: torch.Tensor,
    target: torch.Tensor,
    env_id: torch.Tensor,
    eps: float = 1e-8,
    min_samples: int = 2,
) -> torch.Tensor:
    """
    Compute macro-averaged Pearson correlation across environments.

    Matches eval-time behavior:
    - skip environments with < min_samples
    - skip environments where predictions or targets are constant
    - average Pearson r uniformly across valid environments
    """
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if target.ndim > 1:
        target = target.squeeze(-1)

    pred = pred.float()
    target = target.float()
    env_id = env_id.to(pred.device)

    rs = []
    for env in torch.unique(env_id):
        mask = (env_id == env)
        if int(mask.sum().item()) < min_samples:
            continue

        p = pred[mask]
        t = target[mask]

        p_center = p - p.mean()
        t_center = t - t.mean()
        p_ss = (p_center * p_center).sum()
        t_ss = (t_center * t_center).sum()
        if p_ss <= eps or t_ss <= eps:
            continue

        r = (p_center * t_center).sum() / (p_ss.sqrt() * t_ss.sqrt())
        if torch.isfinite(r):
            rs.append(r.clamp(-1.0, 1.0))

    if len(rs) == 0:
        return torch.tensor(float("nan"), device=pred.device)
    return torch.stack(rs).mean()

class LocalSpearmanCorrLoss(nn.Module):
    """Loss = 1 - Spearman rho (averaged if reduction='mean')."""
    def __init__(self, dim: int = -1, eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.dim, self.eps, self.reduction = dim, eps, reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        rho = torch_spearmanr(pred, target, dim=self.dim, eps=self.eps)
        loss = 1.0 - rho
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class LocalPearsonCorrLoss(nn.Module):
    """
    Local (non-DDP) Pearson loss.
    Loss = 1 - r  (averaged across targets if multi-D).
    """
    def __init__(self, dim=0, eps=1e-8, reduction="mean"):
        super().__init__()
        self.dim, self.eps, self.reduction = dim, eps, reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r = torch_pearsonr(pred, target, dim=self.dim, eps=self.eps)
        loss = 1.0 - r
        if self.reduction == "mean": 
            return loss.mean()
        if self.reduction == "sum":  
            return loss.sum()
        return loss

class GlobalPearsonCorrLoss(nn.Module):
    """
    Pearson correlation loss computed on local batch.
    DDP handles gradient synchronization automatically.
    Works for [N] or [N, D] with correlation over dim=0.
    Loss = 1 - r  (averaged across targets if multi-D).
    """
    def __init__(self, dim=0, eps=1e-8, reduction="mean"):
        super().__init__()
        assert dim == 0, "Global PCC only supports dim=0"
        self.eps, self.reduction = eps, reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.float()
        target = target.float()
        
        # Simply compute local Pearson correlation - DDP syncs gradients
        r = torch_pearsonr(pred, target, dim=0, eps=self.eps)

        loss = 1.0 - r
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss
     
class KTauLoss(nn.Module):
    "Kendall's Tau Correlation"

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.float()
        target = target.float()
        idx = torch.argsort(pred)
        target_a = torch.gather(target, dim=2, index=idx) # sort targets based on ranking of preds

        B, T, N = target_a.shape
        M = N - 1
        refs = target_a[:, :, :-1].unsqueeze(2) # B, T, M, 1
        mask = torch.triu(
            torch.ones(M, N, dtype=torch.bool),
            diagonal=1
        ).view(1, 1, M, N)
        target_a_ex = target_a.unsqueeze(2).expand(B, T, M, N)
        greater = (target_a_ex > refs) & mask
        lesser = (target_a_ex < refs) & mask

        conc = greater.sum(dim=-1).sum(dim=-1, keepdim=True)
        disc = lesser.sum(dim=-1).sum(dim=-1, keepdim=True)
        tau = (conc - disc) / (conc + disc)
        loss = 1.0 - tau
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

class XiLoss(nn.Module):
    "Xi Correlation"

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pass


# =============================================================================
# CONTRASTIVE AND AUXILIARY LOSSES FOR GENERALIZABLE GxE LEARNING
# =============================================================================

def compute_ibs_similarity(g_data: torch.Tensor) -> torch.Tensor:
    """
    Compute Identity-by-State (IBS) similarity between hybrids in a batch.
    
    IBS = 1 - (mean absolute difference / 2)
    For SNPs encoded as 0, 1, 2: max difference is 2.
    
    Args:
        g_data: (batch, n_snps) tensor of genotype data (0, 1, 2 encoded)
    
    Returns:
        (batch, batch) similarity matrix with values in [0, 1]
    """
    # Normalize to [0, 1] range
    g_norm = g_data.float() / 2.0  # Now in [0, 1]
    
    # Compute pairwise L1 distance
    # |g_i - g_j| for each SNP, then mean across SNPs
    diff = g_norm.unsqueeze(1) - g_norm.unsqueeze(0)  # (B, B, n_snps)
    l1_dist = diff.abs().mean(dim=-1)  # (B, B)
    
    # Convert distance to similarity
    similarity = 1.0 - l1_dist
    
    return similarity


def compute_grm_similarity(g_data: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute Genomic Relationship Matrix (GRM) similarity between hybrids.
    
    GRM is the standard in quantitative genetics and accounts for allele frequencies.
    Formula: G = ZZ' / m, where Z is centered and scaled genotype matrix.
    
    This weights rare alleles more heavily (more informative) and is more
    biologically meaningful than IBS for predicting phenotypic similarity.
    
    Args:
        g_data: (batch, n_snps) tensor of genotype data (0, 1, 2 encoded)
        eps: Small constant for numerical stability
    
    Returns:
        (batch, batch) genomic relationship matrix (can be > 1 for inbred, < 0 for unrelated)
    """
    g = g_data.float()
    batch_size, n_snps = g.shape
    
    # Compute allele frequencies from batch (p = mean / 2)
    p = g.mean(dim=0) / 2.0  # (n_snps,)
    
    # Center genotypes: subtract 2p (expected value under HWE)
    g_centered = g - 2.0 * p.unsqueeze(0)  # (batch, n_snps)
    
    # Scale by sqrt(2 * p * (1-p)) - the expected std under HWE
    # This weights rare alleles more heavily
    scale = torch.sqrt(2.0 * p * (1.0 - p) + eps)  # (n_snps,)
    g_scaled = g_centered / scale.unsqueeze(0)  # (batch, n_snps)
    
    # Handle SNPs with no variation (p=0 or p=1)
    # Set their contribution to 0
    valid_snps = (p > eps) & (p < 1.0 - eps)
    g_scaled = g_scaled * valid_snps.float().unsqueeze(0)
    n_valid = valid_snps.sum().clamp_min(1.0)
    
    # GRM = ZZ' / m (m = number of valid SNPs)
    grm = torch.mm(g_scaled, g_scaled.t()) / n_valid
    
    return grm


def compute_env_similarity(
    e_data: torch.Tensor,
    method: str = "cosine",
    e_cat_data: torch.Tensor = None,
    e_cat_cardinalities: list[int] | None = None,
) -> torch.Tensor:
    """
    Compute environment similarity for environment contrastive loss.
    
    If two environments have similar features (weather, soil, etc.),
    hybrids should rank similarly in them.
    
    Args:
        e_data: (batch, n_env_features) tensor of environment features
        method: "cosine" or "correlation"
    
    Returns:
        (batch, batch) environment similarity matrix
    """
    e = e_data.float()

    # Optionally include categorical env fields as one-hot blocks.
    if e_cat_data is not None:
        cat = e_cat_data.long()
        if cat.dim() == 1:
            cat = cat.unsqueeze(1)
        cat_blocks = []
        for j in range(cat.size(1)):
            if e_cat_cardinalities is not None and j < len(e_cat_cardinalities):
                n_classes = max(1, int(e_cat_cardinalities[j]))
            else:
                n_classes = int(cat[:, j].max().item()) + 1 if cat.size(0) > 0 else 1
            ids = cat[:, j].clamp(min=0, max=n_classes - 1)
            cat_blocks.append(F.one_hot(ids, num_classes=n_classes).float())
        if cat_blocks:
            e = torch.cat([e] + cat_blocks, dim=1)
    
    if method == "cosine":
        # L2 normalize then dot product
        e_norm = F.normalize(e, dim=1)
        similarity = torch.mm(e_norm, e_norm.t())
    elif method == "correlation":
        # Center then normalize (Pearson correlation)
        e_centered = e - e.mean(dim=1, keepdim=True)
        e_norm = F.normalize(e_centered, dim=1)
        similarity = torch.mm(e_norm, e_norm.t())
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return similarity


class GenomicContrastiveLoss(nn.Module):
    """
    Learn G embeddings where genetically similar hybrids have similar embeddings.
    
    Uses genetic similarity (GRM or IBS) as soft supervision to encourage the G encoder
    to learn meaningful representations that capture functional similarity.
    
    Improvements over v1:
    - GRM instead of IBS (weights by allele frequency, more biologically meaningful)
    - MSE loss instead of KL (more stable, clearer gradients)
    - Optional correlation target (align correlation structure, not absolute values)
    """
    
    def __init__(
        self, 
        temperature: float = 0.1,
        similarity_type: str = "grm",  # "grm" or "ibs"
        loss_type: str = "mse",  # "mse", "cosine", or "kl"
    ):
        super().__init__()
        self.temperature = temperature
        self.similarity_type = similarity_type
        self.loss_type = loss_type
    
    def forward(
        self, 
        g_embeddings: torch.Tensor,  # (batch, d_model)
        g_data: torch.Tensor = None,  # (batch, n_snps) - raw genotypes for computing similarity
        genetic_similarity: torch.Tensor = None  # (batch, batch) - precomputed similarity
    ) -> torch.Tensor:
        """
        Args:
            g_embeddings: Hybrid embeddings from G encoder (batch, d_model)
            g_data: Raw genotype data to compute similarity (if genetic_similarity not provided)
            genetic_similarity: Precomputed similarity matrix (optional)
        """
        # Compute genetic similarity if not provided
        if genetic_similarity is None:
            if g_data is None:
                raise ValueError("Either g_data or genetic_similarity must be provided")
            if self.similarity_type == "grm":
                genetic_similarity = compute_grm_similarity(g_data)
            else:
                genetic_similarity = compute_ibs_similarity(g_data)
        
        # Compute embedding similarity (cosine similarity)
        g_norm = F.normalize(g_embeddings, dim=1)
        embed_sim = torch.mm(g_norm, g_norm.t())  # (batch, batch), range [-1, 1]
        
        if self.loss_type == "mse":
            # Normalize GRM to similar range as cosine similarity
            # GRM typically ranges from -0.5 to 2+ for inbred lines
            # Shift and scale to roughly [-1, 1]
            if self.similarity_type == "grm":
                # Normalize GRM: shift so diagonal is ~1, off-diagonal centered at ~0
                diag = genetic_similarity.diag().mean()
                target_sim = genetic_similarity / diag.clamp_min(0.1)
                target_sim = target_sim.clamp(-1, 2)  # Clip extreme values
            else:
                # IBS is already in [0, 1], shift to [-1, 1]
                target_sim = 2.0 * genetic_similarity - 1.0
            
            # MSE between embedding similarity and genetic similarity
            # Exclude diagonal (always 1 for both)
            mask = ~torch.eye(embed_sim.size(0), dtype=torch.bool, device=embed_sim.device)
            loss = F.mse_loss(embed_sim[mask], target_sim[mask])
            
        elif self.loss_type == "cosine":
            # Flatten and compute cosine similarity between similarity matrices
            # This aligns the *structure* of relationships, not absolute values
            embed_flat = embed_sim.flatten()
            target_flat = genetic_similarity.flatten()
            
            # Cosine similarity between flattened matrices (want to maximize, so 1 - cosine)
            loss = 1.0 - F.cosine_similarity(
                embed_flat.unsqueeze(0), 
                target_flat.unsqueeze(0)
            ).squeeze()
            
        elif self.loss_type == "kl":
            # Original KL divergence approach (softmax over rows)
            embed_logits = embed_sim / self.temperature
            target_probs = F.softmax(genetic_similarity / self.temperature, dim=1)
            log_probs = F.log_softmax(embed_logits, dim=1)
            loss = -torch.sum(target_probs * log_probs) / g_embeddings.size(0)
            
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss


class EnvironmentContrastiveLoss(nn.Module):
    """
    Learn E embeddings where similar environments have similar embeddings.

    Mirrors GenomicContrastiveLoss:
    - target relationship matrix is computed from raw environment vectors
    - learned relationship matrix is computed from environment embeddings
    - the loss aligns those matrices (default: MSE on off-diagonal entries)

    Args:
        temperature: Softmax temperature for KL mode.
        similarity_method: How to build target env similarity from e_data.
            Supported: "cosine", "correlation".
        loss_type: Alignment objective: "mse", "cosine", or "kl".
    """

    def __init__(
        self,
        temperature: float = 0.5,
        similarity_method: str = "cosine",
        loss_type: str = "mse",
    ):
        super().__init__()
        self.temperature = temperature
        self.similarity_method = similarity_method
        self.loss_type = loss_type

    def forward(
        self,
        e_embeddings: torch.Tensor,            # (batch, d_model)
        e_data: torch.Tensor = None,           # (batch, n_env_features)
        e_cat_data: torch.Tensor = None,       # (batch, n_env_categorical)
        e_cat_cardinalities: list[int] | None = None,
        env_similarity: torch.Tensor = None,   # (batch, batch), optional precomputed target sim
    ) -> torch.Tensor:
        if e_embeddings is None:
            raise ValueError("e_embeddings must be provided")

        if e_embeddings.dim() > 2:
            e_embeddings = e_embeddings.reshape(e_embeddings.size(0), -1)
        if e_embeddings.dim() == 1:
            e_embeddings = e_embeddings.unsqueeze(0)
        e_embeddings = e_embeddings.float()

        if e_embeddings.size(0) < 2:
            return e_embeddings.sum() * 0.0

        # Compute target environment similarity if not precomputed.
        if env_similarity is None:
            if e_data is None:
                raise ValueError("Either e_data or env_similarity must be provided")
            env_similarity = compute_env_similarity(
                e_data,
                method=self.similarity_method,
                e_cat_data=e_cat_data,
                e_cat_cardinalities=e_cat_cardinalities,
            )
        env_similarity = env_similarity.float().to(e_embeddings.device)

        # Learn embedding similarity via cosine similarity.
        e_norm = F.normalize(e_embeddings, dim=1)
        embed_sim = torch.mm(e_norm, e_norm.t())

        if self.loss_type == "mse":
            # Align pairwise structure; ignore diagonal (trivially 1.0).
            mask = ~torch.eye(embed_sim.size(0), dtype=torch.bool, device=embed_sim.device)
            if not mask.any():
                return e_embeddings.sum() * 0.0
            target_sim = env_similarity.clamp(-1.0, 1.0)
            loss = F.mse_loss(embed_sim[mask], target_sim[mask])
        elif self.loss_type == "cosine":
            embed_flat = embed_sim.flatten()
            target_flat = env_similarity.flatten()
            loss = 1.0 - F.cosine_similarity(
                embed_flat.unsqueeze(0),
                target_flat.unsqueeze(0),
            ).squeeze()
        elif self.loss_type == "kl":
            temp = max(self.temperature, 1e-6)
            embed_logits = embed_sim / temp
            target_probs = F.softmax(env_similarity / temp, dim=1)
            log_probs = F.log_softmax(embed_logits, dim=1)
            loss = -torch.sum(target_probs * log_probs) / e_embeddings.size(0)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


class TripletRankingLoss(nn.Module):
    """
    Learn to rank hybrids correctly within environments using triplet loss.
    
    For each anchor hybrid, find a positive (higher yield) and negative (lower yield).
    Penalize if prediction order doesn't match target order.
    
    This is a margin-based alternative to correlation losses.
    """
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        predictions: torch.Tensor,  # (batch,) predicted yields
        targets: torch.Tensor,       # (batch,) true yields
        env_ids: torch.Tensor        # (batch,) environment identifiers
    ) -> torch.Tensor:
        """
        For each sample, find samples in the same env with higher/lower yield.
        Penalize if prediction order doesn't match target order.
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
            
        batch_size = predictions.size(0)
        total_loss = torch.tensor(0.0, device=predictions.device)
        n_triplets = 0
        
        for env in env_ids.unique():
            mask = env_ids == env
            env_pred = predictions[mask]
            env_true = targets[mask]
            n = env_pred.size(0)
            
            if n < 3:
                continue
            
            # Create all valid triplets efficiently
            # For each pair (i, j) where true[i] > true[j], pred[i] should be > pred[j]
            true_diff = env_true.unsqueeze(0) - env_true.unsqueeze(1)  # (n, n)
            pred_diff = env_pred.unsqueeze(0) - env_pred.unsqueeze(1)  # (n, n)
            
            # Mask for valid pairs: true[i] > true[j] + threshold
            valid_pairs = true_diff > 0.2  # At least 0.2 Mg/ha difference
            
            if valid_pairs.sum() == 0:
                continue
            
            # Hinge loss: pred[i] - pred[j] should be > margin when true[i] > true[j]
            violations = F.relu(self.margin - pred_diff[valid_pairs])
            total_loss = total_loss + violations.sum()
            n_triplets += valid_pairs.sum().item()
        
        if n_triplets > 0:
            return total_loss / n_triplets
        else:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)


class RankingConsistencyLoss(nn.Module):
    """
    Encourage consistent hybrid rankings across similar environments.
    
    If Environment A and B are similar (e.g., same location, similar weather),
    then the ranking of hybrids should be similar. This teaches the model
    that rankings should transfer across environments.
    
    Requires pairs of samples from the same hybrid in different environments.
    """
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        predictions: torch.Tensor,  # (batch,) predictions
        targets: torch.Tensor,       # (batch,) true yields
        hybrid_ids: torch.Tensor,    # (batch,) hybrid identifiers
        env_ids: torch.Tensor        # (batch,) environment identifiers
    ) -> torch.Tensor:
        """
        For hybrids appearing in multiple environments, encourage consistent rankings.
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
            
        # Find hybrids that appear in multiple environments
        unique_hybrids = hybrid_ids.unique()
        
        total_loss = torch.tensor(0.0, device=predictions.device)
        n_pairs = 0
        
        for hybrid in unique_hybrids:
            mask = hybrid_ids == hybrid
            if mask.sum() < 2:
                continue
            
            # Get predictions and targets for this hybrid across envs
            h_pred = predictions[mask]
            h_true = targets[mask]
            h_envs = env_ids[mask]
            
            # If same hybrid has different rankings in different envs, 
            # that's GxE - we want to capture it, not penalize it
            # So this loss encourages the MODEL's predictions to match
            # the true pattern (whatever it is)
            
            # Rank predictions should match rank of true values
            pred_ranks = h_pred.argsort().argsort().float()
            true_ranks = h_true.argsort().argsort().float()
            
            # Spearman-like loss on the hybrid's performance across envs
            rank_corr = F.cosine_similarity(
                pred_ranks - pred_ranks.mean(),
                true_ranks - true_ranks.mean(),
                dim=0
            )
            
            # Negative correlation is loss (want high correlation)
            total_loss = total_loss + (1.0 - rank_corr)
            n_pairs += 1
        
        if n_pairs > 0:
            return total_loss / n_pairs
        else:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)


### composite utilities ###
class _CallableLoss(nn.Module):
    """Wrapper to unify nn.Module and callable losses functions"""
    def __init__(self, fn: Union[nn.Module, Callable], expects_env: bool, name: str):
        super().__init__()
        self.fn = fn
        self.expects_env = expects_env
        self.name = name

    def forward(self, pred, target, env_id=None):
        if self.expects_env:
            return self.fn(pred, target, env_id)
        else:
            return self.fn(pred, target)

class CompositeLoss(nn.Module):
    """Combine any number of loss terms.
    Each term has a callable (fn) and a weight (lambda_i).
    Example: CompositeLoss(
        ("mse", nn.MSELoss(), 1.0),
        ("pcc", LocalPearsonCorrLoss(), 0.5))
    """

    def __init__(self, losses: List[Tuple[str, _CallableLoss, float]]):
        super().__init__()
        self.losses = nn.ModuleList([l for _, l, _ in losses])
        self.names = [n for n, _, _ in losses]
        self.weights = [w for _, _, w in losses]

    def forward(self, pred, target, env_id=None):
        total = torch.zeros((), device=pred.device, dtype=pred.dtype)
        parts = {}
        for name, fn, w in zip(self.names, self.losses, self.weights):
            val = fn(pred, target, env_id)
            if not torch.is_tensor(val):
                val = torch.tensor(val, device=pred.device, dtype=pred.dtype)
            total = total + (w * val)
            parts[name] = float(val.detach().item())
        return total, parts

# loss builder
def build_loss(name: str, weights: str = None) -> CompositeLoss:
    """
    name: string like "mse", "pcc", "mse+pcc", "envmse+tau+xi", etc.
    weights: comma-separated weights for each loss term, in order (e.g. "1.0,0.5,0.1")
    returns a CompositeLoss instance
    """
    terms = [s.strip().lower() for s in name.split("+")]
    if weights is not None:
        ws = [float(x) for x in weights.split(",")]
        if len(ws) != len(terms):
            raise ValueError(f"Number of weights ({len(ws)}) does not match number of loss terms ({len(terms)})")
    else:
        ws = [1.0] * len(terms)
    
    def make_one(term: str) -> Tuple[str, _CallableLoss, float]:
        if term == "mse":
            return term, _CallableLoss(nn.MSELoss(), expects_env=False, name=term)
        if term == "pcc":
            return term, _CallableLoss(LocalPearsonCorrLoss(dim=0), expects_env=False, name=term)
        if term == "spearman":
            return term, _CallableLoss(LocalSpearmanCorrLoss(dim=0), expects_env=False, name=term)
        if term == "envmse":
            return term, _CallableLoss(envwise_mse, expects_env=True, name=term)
        if term == "envpcc":
            return term, _CallableLoss(envwise_pcc, expects_env=True, name=term)
        if term == "envspearman":
            return term, _CallableLoss(envwise_spearman, expects_env=True, name=term)
        if term == "ktau":
            return term, _CallableLoss(KTauLoss(), expects_env=False, name=term)
        if term == "xi":
            return term, _CallableLoss(XiLoss(), expects_env=False, name=term)
        if term == "triplet":
            return term, _CallableLoss(TripletRankingLoss(), expects_env=True, name=term)
        raise ValueError(f"Unknown loss term: {term}")
    
    loss_list = []
    for t, w in zip(terms, ws):
        name_t, fn_t = make_one(t)
        loss_list.append((name_t, fn_t, w))
    return CompositeLoss(loss_list)
        
