import torch
import torch.nn as nn
import torch.distributed as dist
from torchsort import soft_rank
from typing import Callable, Union, Tuple, Dict, List

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

def envwise_mse(pred, target, env_id, eps: float = 1e-8):
    """
    Mean of per-environment MSE (with per-env centering).
    Computes loss LOCALLY to maintain gradient flow - DDP handles gradient sync.
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

    # Compute LOCAL sufficient statistics only (no all-reduce)
    count = _accumulate(torch.ones_like(pred_f))
    sy = _accumulate(target_f)

    valid = count > 1
    if not valid.any():
        # fallback to simple MSE
        return torch.mean((pred_f - target_f) ** 2)

    # Compute LOCAL per-env target mean for centering
    mean_y = sy / count.clamp_min(1.0)
    
    # Per-sample env mean
    env_mean_y_local = mean_y[env_id]
    
    # Centered squared error (pred - target)^2
    local_mse = (pred_f - target_f) ** 2
    
    return local_mse.mean()

def envwise_pcc(pred, target, env_id, eps=1e-8):
    """
    Compute Pearson r independently for each environment.
    Computes LOCAL correlation per environment - DDP handles gradient sync.
    Uses Fisher z-transform weighted by sample count for stable averaging.
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

    valid = (count > 1) & (var_x > eps) & (var_y > eps)
    if not valid.any():
        # fallback: use global Pearson correlation on local batch
        r = torch_pearsonr(pred, target)
        if not torch.isfinite(r).all():
            return (pred.sum() * 0.0) + 1.0
        return 1.0 - r

    # Weight by sample count (more samples = more reliable estimate)
    # Use Fisher z-transform for stability, then weighted average
    r_valid = r_per_env[valid].clamp(-0.99999, 0.99999)
    count_valid = count[valid]
    
    z = 0.5 * torch.log((1 + r_valid) / (1 - r_valid))
    weights = count_valid / count_valid.sum()
    z_weighted = (z * weights).sum()
    r_bar = torch.tanh(z_weighted)
    
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
        raise ValueError(f"Unknown loss term: {term}")
    
    loss_list = []
    for t, w in zip(terms, ws):
        name_t, fn_t = make_one(t)
        loss_list.append((name_t, fn_t, w))
    return CompositeLoss(loss_list)
        
