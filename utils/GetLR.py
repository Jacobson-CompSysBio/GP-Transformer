import math

# learning rate decay scheduler (cosine with warmup) -- Thanks Andrej Karpathy
def get_lr(it, 
           warmup_iters, 
           lr_decay_iters, 
           max_lr, 
           min_lr):

    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    
    if it > lr_decay_iters + warmup_iters:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1

    return min_lr + coeff * (max_lr - min_lr)