import torch


def _expand_lambda(lam: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    out = lam.to(device=ref.device, dtype=ref.dtype)
    while out.dim() < ref.dim():
        out = out.unsqueeze(-1)
    return out


@torch.no_grad()
def sample_cmixup_pairs(
    y: torch.Tensor,
    env_id: torch.Tensor,
    prob: float = 0.3,
    alpha: float = 0.2,
    temperature: float = 0.25,
    topk: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Select C-Mixup partners using within-environment target proximity.

    Returns:
        partner_idx: LongTensor [B] partner index for each sample
        lam: FloatTensor [B] mix coefficient for each sample
        active: BoolTensor [B] whether sample participates in C-Mixup
    """
    y = y.view(-1).float()
    env_id = env_id.view(-1).long()
    batch_size = y.numel()
    device = y.device

    partner_idx = torch.arange(batch_size, device=device, dtype=torch.long)
    lam = torch.ones(batch_size, device=device, dtype=torch.float32)
    active = torch.zeros(batch_size, device=device, dtype=torch.bool)

    if batch_size < 2 or prob <= 0.0 or alpha <= 0.0:
        return partner_idx, lam, active

    active = torch.rand(batch_size, device=device) < prob
    if not torch.any(active):
        return partner_idx, lam, active

    # Standardize targets within each environment so "closeness" is env-relative.
    y_std = torch.zeros_like(y)
    for env in env_id.unique():
        mask = env_id == env
        vals = y[mask]
        mu = vals.mean()
        sigma = vals.std(unbiased=False)
        y_std[mask] = (vals - mu) / (sigma + 1e-6)

    idx_all = torch.arange(batch_size, device=device)
    temp = max(float(temperature), 1e-6)
    topk = int(topk) if topk is not None else 0

    for i in idx_all[active]:
        ii = int(i.item())
        cand_mask = (env_id == env_id[ii]) & (idx_all != ii)
        cand = idx_all[cand_mask]
        if cand.numel() == 0:
            active[ii] = False
            continue

        dist = torch.abs(y_std[cand] - y_std[ii])
        if topk > 0 and cand.numel() > topk:
            vals, keep = torch.topk(dist, k=topk, largest=False)
            cand = cand[keep]
            dist = vals

        probs = torch.softmax(-dist / temp, dim=0)
        j = cand[torch.multinomial(probs, num_samples=1)]
        partner_idx[ii] = j[0]

    n_active = int(active.sum().item())
    if n_active > 0:
        beta = torch.distributions.Beta(alpha, alpha)
        lam_active = beta.sample((n_active,)).to(device=device, dtype=torch.float32)
        # Keep each anchor dominant to avoid aggressive interpolation.
        lam_active = torch.maximum(lam_active, 1.0 - lam_active)
        lam[active] = lam_active

    return partner_idx, lam, active


def mix_tensor(x: torch.Tensor, partner_idx: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    x_partner = x[partner_idx]
    if x.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        keep_prob = _expand_lambda(lam, x).to(dtype=torch.float32)
        mask = torch.rand_like(keep_prob, dtype=torch.float32) < keep_prob
        return torch.where(mask, x, x_partner)
    lam_exp = _expand_lambda(lam, x)
    return lam_exp * x + (1.0 - lam_exp) * x_partner


def mix_batch_inputs(xb: dict, partner_idx: torch.Tensor, lam: torch.Tensor) -> dict:
    mixed = {}
    for key, value in xb.items():
        if isinstance(value, torch.Tensor):
            mixed[key] = mix_tensor(value, partner_idx, lam)
        else:
            mixed[key] = value
    return mixed


def mix_targets(y: torch.Tensor, partner_idx: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    y_partner = y[partner_idx]
    lam_exp = _expand_lambda(lam, y)
    return lam_exp * y + (1.0 - lam_exp) * y_partner
