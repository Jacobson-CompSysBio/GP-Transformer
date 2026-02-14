import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sample-weighted MSE with weights normalized to mean 1 in-batch."""
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if target.ndim > 1:
        target = target.squeeze(-1)
    w = weights.float()
    w = w / w.mean().clamp_min(eps)
    se = (pred.float() - target.float()) ** 2
    return (w * se).mean()


@dataclass
class ShiftWeightConfig:
    max_train_samples: int = 50000
    max_test_samples: int = 50000
    use_genotype: bool = False
    genotype_marker_dim: int = 512
    clip_min: float = 0.1
    clip_max: float = 10.0
    power: float = 1.0
    seed: int = 1


def _sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)


def _to_float_matrix(x) -> np.ndarray:
    # Works for pd.DataFrame / np.ndarray.
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")
    return arr


def _build_domain_matrix(
    train_ds,
    test_ds,
    *,
    use_genotype: bool,
    genotype_marker_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_tr_e = _to_float_matrix(train_ds.e_data)
    x_te_e = _to_float_matrix(test_ds.e_data)

    if not use_genotype:
        return x_tr_e, x_te_e

    # Use raw dosage if available; otherwise fall back to g_data.
    g_tr = _to_float_matrix(getattr(train_ds, "g_raw_dosage", train_ds.g_data))
    g_te = _to_float_matrix(getattr(test_ds, "g_raw_dosage", test_ds.g_data))

    if genotype_marker_dim > 0 and g_tr.shape[1] > genotype_marker_dim:
        rng = np.random.default_rng(seed)
        cols = np.sort(rng.choice(g_tr.shape[1], size=genotype_marker_dim, replace=False))
        g_tr = g_tr[:, cols]
        g_te = g_te[:, cols]

    x_tr = np.concatenate([x_tr_e, g_tr], axis=1)
    x_te = np.concatenate([x_te_e, g_te], axis=1)
    return x_tr, x_te


def build_train_covariate_shift_weights(
    train_ds,
    *,
    data_path: Optional[str] = None,
    cfg: ShiftWeightConfig,
    ref_ds=None,
    ref_split: str = "test",
    ref_kwargs: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Fit a train-vs-test domain classifier and return density-ratio style
    importance weights for TRAIN samples only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if ref_ds is None:
        if data_path is None:
            raise ValueError("data_path must be provided when ref_ds is None.")
        from utils.dataset import GxE_Dataset

        ds_kwargs = dict(
            split=ref_split,
            data_path=data_path,
            residual=bool(getattr(train_ds, "residual_flag", False)),
            scaler=train_ds.scaler,
            y_scalers=train_ds.label_scalers if train_ds.scale_targets else None,
            scale_targets=train_ds.scale_targets,
            g_input_type=train_ds.g_input_type,
            marker_stats=getattr(train_ds, "marker_stats", None),
        )
        if ref_kwargs:
            ds_kwargs.update(ref_kwargs)
        ref_ds = GxE_Dataset(**ds_kwargs)

    x_tr, x_ref = _build_domain_matrix(
        train_ds,
        ref_ds,
        use_genotype=cfg.use_genotype,
        genotype_marker_dim=cfg.genotype_marker_dim,
        seed=cfg.seed,
    )

    rng = np.random.default_rng(cfg.seed)
    n_fit = min(
        int(cfg.max_train_samples),
        int(cfg.max_test_samples),
        x_tr.shape[0],
        x_ref.shape[0],
    )
    idx_tr_fit = _sample_indices(x_tr.shape[0], n_fit, rng)
    idx_ref_fit = _sample_indices(x_ref.shape[0], n_fit, rng)

    x_fit = np.concatenate([x_tr[idx_tr_fit], x_ref[idx_ref_fit]], axis=0)
    y_fit = np.concatenate(
        [np.zeros(n_fit, dtype=np.int64), np.ones(n_fit, dtype=np.int64)],
        axis=0,
    )

    scaler = StandardScaler()
    x_fit_s = scaler.fit_transform(x_fit)
    clf = LogisticRegression(
        max_iter=300,
        C=1.0,
        solver="lbfgs",
        n_jobs=1,
    )
    clf.fit(x_fit_s, y_fit)

    # Density ratio proxy with balanced-domain prior:
    # w(x) ~= p(test|x) / p(train|x)
    x_tr_s = scaler.transform(x_tr)
    p_test = clf.predict_proba(x_tr_s)[:, 1].astype(np.float64)
    p_test = np.clip(p_test, 1e-5, 1.0 - 1e-5)
    w = p_test / (1.0 - p_test)

    if cfg.power != 1.0:
        w = np.power(w, float(cfg.power))

    w = np.clip(w, float(cfg.clip_min), float(cfg.clip_max))
    w = w / max(1e-12, float(np.mean(w)))
    w_t = torch.tensor(w, dtype=torch.float32)

    auc = roc_auc_score(y_fit, clf.predict_proba(x_fit_s)[:, 1])
    stats = {
        "domain_auc_fit": float(auc),
        "weights_min": float(w.min()),
        "weights_max": float(w.max()),
        "weights_mean": float(w.mean()),
        "weights_std": float(w.std(ddof=0)),
        "train_samples_for_fit": float(n_fit),
        "reference_samples_for_fit": float(n_fit),
        "feature_dim": float(x_fit.shape[1]),
    }
    if ref_ds is not None:
        stats["reference_split"] = str(ref_split)
    return w_t, stats


class GroupDROObjective:
    """
    Lightweight GroupDRO objective over per-sample losses.
    q update uses global (all-rank) detached group losses when DDP is enabled.
    """

    def __init__(self, num_groups: int, step_size: float = 0.01, eps: float = 1e-12):
        self.num_groups = int(num_groups)
        self.step_size = float(step_size)
        self.eps = float(eps)
        self.q = torch.ones(self.num_groups, dtype=torch.float32) / max(1, self.num_groups)

    def to(self, device: torch.device):
        self.q = self.q.to(device=device)
        return self

    def _global_group_means(self, loss_sum: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            loss_sum_g = loss_sum.detach().clone()
            count_g = count.detach().clone()
            dist.all_reduce(loss_sum_g, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_g, op=dist.ReduceOp.SUM)
        else:
            loss_sum_g = loss_sum.detach()
            count_g = count.detach()
        return loss_sum_g / count_g.clamp_min(1.0)

    def __call__(
        self,
        sample_losses: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if sample_losses.ndim > 1:
            sample_losses = sample_losses.squeeze(-1)
        group_ids = group_ids.long()
        if sample_losses.numel() != group_ids.numel():
            raise ValueError(
                f"sample_losses and group_ids must align, got "
                f"{sample_losses.numel()} vs {group_ids.numel()}"
            )

        device = sample_losses.device
        if self.q.device != device:
            self.to(device)

        loss_sum = torch.zeros(self.num_groups, device=device, dtype=sample_losses.dtype)
        count = torch.zeros(self.num_groups, device=device, dtype=sample_losses.dtype)
        loss_sum = loss_sum.scatter_add(0, group_ids, sample_losses)
        count = count.scatter_add(0, group_ids, torch.ones_like(sample_losses))
        group_mean_local = loss_sum / count.clamp_min(1.0)

        # Update q with detached global means for stability under DDP.
        with torch.no_grad():
            group_mean_global = self._global_group_means(loss_sum, count)
            valid = count > 0
            if valid.any():
                self.q[valid] = self.q[valid] * torch.exp(self.step_size * group_mean_global[valid])
                self.q = self.q / self.q.sum().clamp_min(self.eps)

        robust = (self.q.detach() * group_mean_local).sum()
        stats = {
            "group_dro_q_max": float(self.q.max().item()),
            "group_dro_q_min": float(self.q.min().item()),
            "group_dro_groups_in_batch": float((count > 0).sum().item()),
        }
        return robust, stats
