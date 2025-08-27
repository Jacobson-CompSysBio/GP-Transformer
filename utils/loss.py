import torch
import torch.nn as nn
import torch.distributed as dist

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

class PearsonCorrLoss(nn.Module):
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
    
class BothLoss(nn.Module):
    "sum of global pcc and mse loss"

    def __init__(self, alpha: float=0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.pcc = GlobalPearsonCorrLoss(reduction="mean")
        self.alpha = alpha

    def forward(self, pred, target):
        return (self.alpha * self.mse(pred, target)) + ((1-self.alpha) * self.pcc(pred, target))

# ---- tiny factory: only two options (global PCC, MSE) ----
def build_loss(name: str,
               alpha: float = 0.5):
    """
    name is either mse or pcc,
    alpha is the loss weighting for mse
    """
    n = name.lower()
    if n == "mse":
        return nn.MSELoss()
    if n == "pcc":
        return GlobalPearsonCorrLoss()
    if n == "both":
        return BothLoss(alpha=alpha)
    raise ValueError(f"Unknown loss: {name} (expected 'mse', 'pcc', or 'both')")