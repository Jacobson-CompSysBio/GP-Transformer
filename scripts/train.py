# imports
import time, os, json, random, math, argparse, sys
from pathlib import Path
from dotenv import load_dotenv
from contextlib import nullcontext
from tqdm import tqdm
import subprocess
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split

# add parent directory (one level up) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.dataset import *
from utils.model import *
from utils.GetLR import get_lr

load_dotenv()
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

def extract_master_addr():
    try:
        # Use scontrol to get the hostname of the first node
        nodelist = os.environ["SLURM_NODELIST"]
        node = subprocess.check_output(
            ["scontrol", "show", "hostname", nodelist]
        ).decode().splitlines()[0]
        return node
    except Exception as e:
        print(f"[WARN] Failed to extract master address from SLURM_NODELIST: {e}")
        return "localhost"

def setup_ddp():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))

    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    
    return torch.device("cuda:0"), local_rank, rank, world_size


def cleanup_ddp():
    """clean up torch distributed backend"""
    dist.destroy_process_group()

def is_main(rank) -> bool:
    """check if current process is main"""
    return rank == 0

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


### main ###
def main():
    # setup
    args = parse_args()
    device, local_rank, rank, world_size = setup_ddp()

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
    total_iters = args.num_epochs * batches_per_epoch
    warmup_iters = batches_per_epoch
    lr_decay_iters = total_iters
    max_lr, min_lr = (0.1 * args.lr), (0.01 * args.lr) 
    max_iters = total_iters
    max_epochs = args.num_epochs
    eval_interval = batches_per_epoch
    early_stop = 10

    ### wandb logging ###
    if is_main(rank):
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
        )
    
    # initialize training states
    best_val_loss, last_improved = float("inf"), 0
    iter_num = 0
    t0 = time.time()

    ### training loop ###
    for epoch_num in range(max_epochs):
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

            # double check that ddp accumulates gradients 
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
        pbar.close()

        ### evaluation ###
        with torch.no_grad():
            train_loss_accum = 0
            val_loss_accum = 0
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
                }, f"checkpoints/checkpoint_{epoch_num:04d}.pt")
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

    # finish + clean up
    if is_main(rank):
        wandb.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()