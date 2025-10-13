import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Callable, Union, Tuple, Dict, List

### per-env losses ###
def envwise_mse(pred, target, env_id):
    """
    pred: [B, 1] or [B] (float)
    target: [B, 1] or [B] (float)
    env_id: [B] (long) -> same envs share same id

    returns a scalar loss (mean(MSE over envs in batch)) 
    """

    #  squeeze preds, targets if they are [B, 1]
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if target.ndim > 1:
        target = target.squeeze(-1)
    
    loss_acc = torch.zeros((), device=pred.device)
    count = 0
    for env in torch.unique(env_id):
        mask = (env_id == env)
        if int(mask.sum()) < 2:
            continue
        px = pred[mask]
        tx = target[mask]
        px = px - px.mean()
        tx = tx - tx.mean()
        loss_acc = loss_acc + torch.mean((px - tx) ** 2)
        count += 1
    if count == 0:
        return torch.zeros((), device = pred.device, dtype=pred.dtype, requires_grad=True)
    return loss_acc / count

def envwise_pcc(pred, target, env_id):
    """
    pred: [B, 1] or [B] (float)
    target: [B, 1] or [B] (float)
    env_id: [B] (long) -> same envs share same id

    returns a scalar loss (1 - mean(PCC over envs in batch)) 
    """

    #  squeeze preds, targets if they are [B, 1]
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if target.ndim > 1:
        target = target.squeeze(-1)

    unique_envs = torch.unique(env_id)
    pccs = []
    for env in unique_envs:
        mask = (env_id == env)
        
        # don't compute pcc if only one sample in env
        n = int(mask.sum())
        if n < 2:
            continue
        
        # only keep samples from this env
        x = pred[mask]
        y = target[mask]

        # calculate pcc
        x = x - x.mean()
        y = y - y.mean()
        sx = torch.sqrt((x * x).sum())
        sy = torch.sqrt((y * y).sum())
        denom = sx * sy

        # if denom ~ 0, skip (constant group)
        if torch.any(torch.isclose(denom, torch.zeros_like(denom))):
            continue
        
        pcc = (x * y).sum() / denom
        pccs.append(pcc)

    # no vaild groups in batch; return 0 loss contribution 
    if len(pccs) == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype, requires_grad=True)
    
    return 1.0 - torch.stack(pccs).mean()

### other losses ### 
def torch_pearsonr(pred: torch.Tensor, target: torch.Tensor, dim=0, eps=1e-8):
    """
    pred, target: [N] or [N, D]; set dim=0 for batch-wise correlation.
    Returns r with shape [] if vectors, or [D] if multi-dim features.
    Non-DDP helper (local only). Kept for completeness.
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
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

class GlobalPearsonCorrLoss(nn.Module):
    """
    Global Pearson (DDP-wide) over the current step.
    Works for [N] or [N, D] with correlation over dim=0.
    Loss = 1 - r  (averaged across targets if multi-D).
    """
    def __init__(self, dim=0, eps=1e-8, reduction="mean"):
        super().__init__()
        assert dim == 0, "Global PCC only supports dim=0"
        self.eps, self.reduction = eps, reduction

    def _sufficient_stats(self, x, y):
        # [N] or [N, D] -> keep feature dim, reduce batch dim
        if x.dim() == 1:
            x = x[:, None]
            y = y[:, None]
        n   = torch.tensor([x.size(0)], device=x.device, dtype=torch.float64)
        sx  = x.sum(dim=0, dtype=torch.float64)
        sy  = y.sum(dim=0, dtype=torch.float64)
        sxx = (x * x).sum(dim=0, dtype=torch.float64)
        syy = (y * y).sum(dim=0, dtype=torch.float64)
        sxy = (x * y).sum(dim=0, dtype=torch.float64)
        return n, sx, sy, sxx, syy, sxy

    def _all_reduce_stats(self, stats):
        if not (dist.is_available() and dist.is_initialized()):
            return stats
        reduced = []
        for t in stats:
            tc = t.clone()
            dist.all_reduce(tc, op=dist.ReduceOp.SUM)
            reduced.append(tc)
        return tuple(reduced)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.float()
        target = target.float()

        n, sx, sy, sxx, syy, sxy = self._sufficient_stats(pred, target)
        n, sx, sy, sxx, syy, sxy = self._all_reduce_stats((n, sx, sy, sxx, syy, sxy))

        n_scalar = n.item()
        cov  = sxy - (sx * sy) / n_scalar
        varx = sxx - (sx * sx) / n_scalar
        vary = syy - (sy * sy) / n_scalar

        r = (cov / (varx.clamp_min(self.eps).sqrt() * vary.clamp_min(self.eps).sqrt())).float()
        r = r.clamp(-1.0, 1.0)

        loss = 1.0 - r
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss
     
class KTauLoss(nn.Module):
    "Kendall's Tau Correlation"

    def __init__(self, reduction="mean"):
        super().__init_()
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
        super().__init_()
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
        if term == "envmse":
            return term, _CallableLoss(envwise_mse, expects_env=True, name=term)
        if term == "envpcc":
            return term, _CallableLoss(envwise_pcc, expects_env=True, name=term)
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
        


