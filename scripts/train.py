# imports
import time, os, json, random, math, argparse
from pathlib import Path
from contextlib import nullcontext
from tqdm import tqdm
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split

from ..utils.dataset import *
from ..utils.model import *
from ..utils.GetLR import get_lr

def setup_ddp():
    """initialize torch distributed backend with srun vars"""
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    """clean up torch distributed backend"""
    dist.destroy_process_group()

def is_main(rank) -> bool:
    """check if current process is main"""
    return rank == 0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--num_iters",   type=int, default=50_000)
    p.add_argument("--seed",        type=int, default=1)
    p.add_argument("--project",     default="gxe-ddp")
    p.add_argument("--run_name",    default=None)
    return p.parse_args()


### main ###
def main():
    # setup
    args = parse_args()
    rank, world_size = setup_ddp()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

    # reproducibility 
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # data (samplers are needed for DDP)
    gxe = GxE_Dataset(split="train")
    gxe_train, gxe_val = random_split(gxe, 
                                      [int(len(gxe)*0.8), len(gxe)-int(len(gxe)*0.8)])
    train_sampler = DistributedSampler(gxe_train, shuffle=True)
    val_sampler = DistributedSampler(gxe_val, shuffle=False)
    train_loader = DataLoader(
        gxe_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_loader = DataLoader(
        gxe_val, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        pin_memory=True)

    # set up config
    config = TransformerConfig(block_size=len(gxe_train[0][0]['g_data']))
    model = GxE_Transformer(config=config).to(device)
    model = DDP(model, device_ids=[device])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_function = torch.nn.MSELoss()


    # other options
    batches_per_epoch = len(train_loader)
    batches_per_eval = len(val_loader)
    warmup_iters = batches_per_epoch
    lr_decay_iters = args.num_iters
    max_lr, min_lr = 1e-3, 1e-7
    max_iters = args.num_iters
    eval_interval = batches_per_epoch
    early_stop = 10

    ### wandb logging ###
    if is_main(rank):
        wandb.init(
            project=args.project,
            name=args.run_name,
            config=vars(args)
        )
        wandb.watch(model, log="gradients", log_freq=eval_interval)
    
    # initialize training states
    best_val_loss, last_improved = float("inf"), 0
    iter_num, epoch_num = 0, 0
    t0 = time.time()

    ### training loop ###
    while iter_num < max_iters:
        train_sampler.set_epoch(epoch_num)
        model.train()

        pbar = tqdm(total=eval_interval,
                    desc=f"Rank {rank} Train",
                    disable=not is_main(rank))
        
        # training steps
        for xb, yb in train_loader:
            for k, v in xb.items():
                xb[k] = v.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # fwd/bwd pass
            logits = model(xb)
            loss = loss_function(logits, yb)
            if torch.isnan(loss):
                raise RuntimeError("Loss is NaN, stopping training.")

            # apply learning rate schedule              
            lr = get_lr(iter_num,
                        warmup_iters,
                        lr_decay_iters,
                        max_lr,
                        min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # log wandb
            if is_main(rank):
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                    "iter_num": iter_num,
                })
            iter_num += 1
            pbar.update(1)
            if pbar.n == pbar.total:
                break
        pbar.close

        ### evaluation ###
        with torch.no_grad():
            for (xbt, ybt), (xbv, ybv) in zip(train_loader, val_loader):
                for k, v in xbt.items():
                    xbt[k] = v.to(device, non_blocking=True)
                ybt = ybt.to(device, non_blocking=True)
                for k, v in xbv.items():
                    xbv[k] = v.to(device, non_blocking=True)
                ybv = ybv.to(device, non_blocking=True)

                train_loss_accum += loss_function(model(xbt), ybt)
                val_loss_accum += loss_function(model(xbv), ybv)
                if train_loss_accum.isnan() or val_loss_accum.isnan():
                    raise RuntimeError("Loss is NaN during evaluation, stopping training.")
        
        # aggregate losses over all ranks
        dist.all_reduce(train_loss_accum)
        dist.all_reduce(val_loss_accum)

        train_loss = (train_loss_accum / batches_per_eval / world_size).item()
        val_loss = (val_loss_accum / batches_per_eval / world_size).item()

        # log eval / early stop (only rank 0)
        if is_main(rank):
            wandb.log({
                "val_loss": val_loss,
                "train_loss_epoch": train_loss,
                "epoch": epoch_num,
                "iter": iter_num
            }, step=iter_num)

            if val_loss < best_val_loss:
                best_val_loss, last_improved = val_loss, 0
                torch.save({
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch_num,
                    "val_loss": val_loss
                }, f"checkpoint_{epoch_num:04d}.pt")
                print(f"*** validation loss improved: {best_val_loss:.4e} ***")
            else:
                last_improved += 1
                print(f"Validation has not improved in {last_improved} steps")
            if last_improved > early_stop:
                print(f"*** no improvement for {early_stop} steps, stopping ***")
                break

            elapsed = (time.time() - t0) / 60
            print(f"[Epoch {epoch_num}] train={train_loss:.4e} | "
                  f"val={val_loss:.4e} | best_val={best_val_loss:.4e} | "
                  f"elapsed={elapsed:.2f}m")
        epoch_num += 1

    # finish + clean up
    if is_main(rank):
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()