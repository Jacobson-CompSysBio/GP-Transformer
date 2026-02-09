import math

# learning rate schedule: linear warmup -> cosine decay to a floor
def get_lr(it, warmup, total, max_lr, min_lr):
    """
    it: current step (int)
    warmup: steps for linear warmup
    decay_iters: steps for cosine decay after warmup
    max_lr:      peak learning rate
    min_lr:      floor learning rate (not ~0; keep a few % of max_lr)
    """
    if it < warmup:
        return max_lr * it / max(1, warmup)
    t = (it - warmup) / max(1, total - warmup)
    t = min(t, 1.0)  # clamp so LR never rebounds after cosine completes
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))
