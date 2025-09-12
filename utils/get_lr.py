import math

# learning rate decay scheduler (linear warmup + cosine decay)
def get_lr(it, warmup_iters, decay_iters, max_lr, min_lr):
    if it < warmup_iters:
        return max_lr * it / max(1, warmup_iters)
    if it >= warmup_iters + decay_iters:
        return min_lr
    # cosine from max_lr -> min_lr over `decay_iters` steps after warmup
    ratio = (it - warmup_iters) / max(1, decay_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)