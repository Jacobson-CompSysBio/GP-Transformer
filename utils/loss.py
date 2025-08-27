import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

def torch_pearsonr(pred: torch.Tensor, target: torch.Tensor, dim=0, eps=1e-8):
    """
    pred, target: shape [N] or [N, D] (set dim=0 for batch-wise correlation)
    returns r with shape [] if vectors, or [D] if multi-dim features
    """
    pred = pred.float()
    target = target.float()

    # center the data
    pred = pred - pred.mean(dim=dim, keepdim=True)
    target = target - target.mean(dim=dim, keepdim=True)

    # calculate variance for pred and target
    v_pred = pred.pow(2).sum(dim=dim, keepdim=False)
    v_target = target.pow(2).sum(dim=dim, keepdim=False)

    # use centered pred and target to calculate cov
    cov = (pred * target).sum(dim=dim, keepdim=False)

    # calculate r = cov / sqrt(v_pred * v_target)
    r = cov / (v_pred.clamp_min(eps).sqrt() * v_target.clamp_min(eps).sqrt())
    return r

# non-DDP version
class PearsonCorrLoss(nn.Module):
    """
    Loss = 1 - mean_ij r_ij (to maximize correlation, minimize 1-r)
    If y is multi-target, the loss is computed as the mean over all targets.
    """
    def __init__(self, dim=0, eps=1e-8, reduction="mean"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        r = torch_pearsonr(pred, target, dim=self.dim, eps=self.eps)
        loss = 1.0 - r
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss # reduction == 'none'

# DDP version
class GlobalPearsonCorrLoss(nn.Module):
    """
    Computes PCC across all samples on all DDP ranks in the current step
    Works for shapes, [N] or [N, D] with correlation over dim=0
    """
    def __init__(self, dim=0, eps=1e-8, reduction="mean"):
        super().__init__()
        assert dim == 0, "Global PCC only supports dim=0"
        self.eps = eps
        self.reduction = reduction
    
    @torch.no_grad()
    def _sufficient_stats(self, x, y):
        # shape: [N] or [N, D] --> flatten batch dim only
        if x.dim() == 1:
            x = x[:, None]
            y = y[:, None]
        
        n = torch.tensor([x.size(0)], device=x.device, dtype=torch.float64)
        sx = x.sum(dim=0, dtype=torch.float64)
        sy = y.sum(dim=0, dtype=torch.float64)
        sxx = (x * x).sum(dim=0, dtype=torch.float64)
        syy = (y * y).sum(dim=0, dtype=torch.float64)
        sxy = (x * y).sum(dim=0, dtype=torch.float64)

        return n, sx, sy, sxx, syy, sxy

    def _all_reduce_stats(self, stats):
        if not dist.is_available() or not dist.is_initialized():
            return stats
        
        red = []
        for t in stats:
            t = t.clone()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            red.append(t)
        return tuple(red)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.float()
        target = target.float()

        n, sx, sy, sxx, syy, sxy = self._sufficient_stats(pred, target)
        n, sx, sy, sxx, syy, sxy = self._all_reduce_stats((n, sx, sy, sxx, syy, sxy))

        # compute r from sums
        # r = cov / (std_x * std_y); cov = (sxy - sx*sy/n)
        n = n.item() # convert to scalar
        sxsy_over_n = (sx * sy) / n
        cov = sxy - sxsy_over_n
        varx = sxx - (sx * sx) / n
        vary = syy - (sy * sy) / n

        r = (cov / (varx.clamp_min(self.eps).sqrt() * vary.clamp_min(self.eps).sqrt())).float()
        r = r.clamp(-1, 1)

        loss = 1.0 - r
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss # reduction == 'none'