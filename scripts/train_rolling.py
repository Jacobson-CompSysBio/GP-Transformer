# imports
import time, os, json, random, math, argparse, sys
import shutil
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
from models.model import *
from models.config import Config
from utils.get_lr import get_lr
from utils.loss import build_loss, GlobalPearsonCorrLoss
from utils.utils import parse_args, make_run_name

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

    local_rank = 0
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

    return torch.device(f"cuda:{local_rank}"), local_rank, rank, world_size

def cleanup_ddp():
    """clean up torch distributed backend"""
    dist.destroy_process_group()

def is_main(rank) -> bool:
    """check if current process is main"""
    return rank == 0

### main ###
def main():
    # setup
    args = parse_args()
    device, local_rank, rank, world_size = setup_ddp()
    wandb_run_name = make_run_name(args)

    if is_main(rank):
        run_ckpt_dir = Path("checkpoints") / wandb_run_name
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # reproducibility 
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # define rolling-year folds
    # train <= T; val = T+1
    folds = [(T, T+1) for T in range(2014, 2023)]

    # store fold results
    from collections import defaultdict
    fold_records = []

    # wandb setup
    if is_main(rank):
        wandb_run_name = make_run_name(args)
        run = wandb.init(
            project="gxe-transformer-rolling",
            entity=os.getenv("WANDB_ENTITY"),
            name=wandb_run_name,
            resume="allow",
        )
        # expose run id for eval.py to resume
        os.environ["WANDB_RUN_ID"] = run.id

        # config snapshot
        wandb.config.update({
            "loss": args.loss,
            "alpha": args.alpha},
            allow_val_change=True)

    for fold_idx, (train_year_max, val_year) in enumerate(folds, start=1):
        if is_main(rank):
            print(f"\n=== Fold {fold_idx}/{len(folds)}: train <= {train_year_max}, val = {val_year} ===")
            # per-year step metrics + namespaces
            run.define_metric(f"fold/{val_year}/iter")
            run.define_metric(f"fold/{val_year}/epoch")
            run.define_metric(f"fold/{val_year}/*", step_metric=f"fold/{val_year}/iter")
            run.define_metric(f"fold/{val_year}/epoch/*", step_metric=f"fold/{val_year}/epoch")

        # local fold iteration count
        iter_in_fold = 0

        # create datasets for this fold
        train_ds = GxE_Dataset(
            split="train",
            data_path='data/maize_data_2014-2023_vs_2024/',
            index_map_path='data/maize_data_2014-2023_vs_2024/location_2014_2023.csv',
            scaler=None,
            train_year_max=train_year_max
        )
        scaler = train_ds.scaler
        val_ds = GxE_Dataset(
            split="val",
            data_path="data/maize_data_2014-2023_vs_2024/",
            index_map_path="data/maize_data_2014-2023_vs_2024/location_2014_2023.csv",
            scaler=scaler,
            val_year=val_year
        )

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=args.batch_size, 
            sampler=val_sampler,
            pin_memory=True)

        # set up config
        config = Config(block_size=len(train_ds[0][0]['g_data']),
                        n_head=args.heads,
                        n_g_layer=args.g_layers,
                        n_ld_layer=args.ld_layers,
                        n_mlp_layer=args.mlp_layers,
                        n_gxe_layer=args.gxe_layers,
                        n_embd=args.emb_size,
                        dropout=args.dropout)
        model = GxE_Transformer(g_enc=args.g_enc,
                                e_enc=args.e_enc,
                                ld_enc=args.ld_enc,
                                gxe_enc=args.gxe_enc,
                                moe=args.moe,
                                config=config).to(device)
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        loss_function = build_loss(args.loss, args.alpha)
        mse_loss_log = nn.MSELoss(reduction="mean")
        pcc_loss_log = GlobalPearsonCorrLoss(reduction="mean")

        # other options
        batches_per_epoch = len(train_loader)
        total_iters = args.num_epochs * batches_per_epoch
        warmup_iters = batches_per_epoch  # warmup for 1 epoch - should be enough
        lr_decay_iters = total_iters * .5
        max_lr, min_lr = (args.lr), (0.001 * args.lr)
        max_epochs = args.num_epochs
        eval_interval = batches_per_epoch
        early_stop = args.early_stop

        # initialize training states
        best_val_loss, last_improved = float("inf"), 0
        best_fold_snapshot = None
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
                yb = yb.to(device, non_blocking=True).float()

                # fwd/bwd pass
                logits = model(xb)
                loss = loss_function(logits, yb)

                # sanity checks
                assert any(p.requires_grad for p in model.parameters()), "All params have requires_grad=False!"
                assert logits.is_floating_point(), f"logits dtype must be floating (got {logits.dtype})"
                assert logits.requires_grad, "logits.requires_grad=False (check encoders/final head for detach/int/argmax/round)"
                assert loss.requires_grad, "loss.requires_grad=False (computed under no_grad or logits detached?)"
            
                # also check final head weights are trainable
                fw = model.module.final_layer.weight if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.final_layer.weight
                assert fw.requires_grad, "final_layer weights are frozen"

                if torch.isnan(loss):
                    raise RuntimeError("Loss is NaN, stopping training.")

                # compute losses for wandb logging 
                train_mse_val = None
                train_pcc_loss_val = None
                if args.loss == "both":
                    with torch.no_grad():
                        train_mse_val = mse_loss_log(logits, yb)
                        train_pcc_loss_val = pcc_loss_log(logits, yb)

                # apply learning rate schedule              
                lr = get_lr(iter_in_fold,
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
                    log_payload = {
                        f"fold/{val_year}/iter": iter_in_fold,
                        f"fold/{val_year}/train_loss": loss.item(),
                        f"fold/{val_year}/learning_rate": lr,
                    }
                    if args.loss == "both":
                        log_payload[f"fold/{val_year}/mse_iter"] = float(train_mse_val.item())
                        log_payload[f"fold/{val_year}/pcc_loss_iter"] = float(train_pcc_loss_val.item())
                    wandb.log(log_payload)
                iter_in_fold += 1
                pbar.update(1)
            pbar.close()

            ### evaluation ###
            with torch.no_grad():
                model.eval()

                def eval_loader(loader):
                    total_loss = torch.tensor(0.0, device=device)
                    mse_acc = torch.tensor(0.0, device=device)
                    pcc_acc = torch.tensor(0.0, device=device)
                    n_batches = 0
                    for xb, yb in loader:
                        xb = {k: v.to(device, non_blocking=True) for k, v in xb.items()}
                        yb = yb.to(device, non_blocking=True).float()
                        preds = model(xb)
                        total_loss += loss_function(preds, yb)
                        mse_acc += mse_loss_log(preds, yb)
                        pcc_acc += pcc_loss_log(preds, yb)
                        n_batches += 1

                    return total_loss, mse_acc, pcc_acc, n_batches

                train_loss_accum, train_mse_accum, train_pcc_loss_accum, n_train = eval_loader(train_loader)
                val_loss_accum, val_mse_accum, val_pcc_loss_accum, n_val = eval_loader(val_loader)

            # aggregate losses over all ranks
            dist.all_reduce(train_loss_accum)
            dist.all_reduce(val_loss_accum)
            for t in (train_mse_accum, train_pcc_loss_accum, val_mse_accum, val_pcc_loss_accum):
                dist.all_reduce(t)

            train_loss = (train_loss_accum / n_train / world_size).item()
            val_loss = (val_loss_accum / n_val / world_size).item()
            train_mse = (train_mse_accum / n_train / world_size).item()
            train_pcc_loss = (train_pcc_loss_accum / n_train / world_size).item()
            val_mse = (val_mse_accum / n_val / world_size).item()
            val_pcc_loss = (val_pcc_loss_accum / n_val / world_size).item()

            # log eval / early stop (only rank 0)
            if is_main(rank):
                wandb.log({
                    f"fold/{val_year}/epoch": epoch_num,
                    f"fold/{val_year}/epoch/train_loss": train_loss,
                    f"fold/{val_year}/epoch/val_loss": val_loss,
                    f"fold/{val_year}/epoch/train_mse": train_mse,
                    f"fold/{val_year}/epoch/train_pcc": train_pcc_loss,
                    f"fold/{val_year}/epoch/val_mse": val_mse,
                    f"fold/{val_year}/epoch/val_pcc": val_pcc_loss,
                }) 

                if val_loss < best_val_loss:
                    best_val_loss, last_improved = val_loss, 0
                    best_fold_snapshot = {
                        "fold": fold_idx,
                        "train_le": train_year_max,
                        "val_y": val_year,
                        "best_val_loss": best_val_loss,
                        "best_val_mse": val_mse,
                        "best_val_pcc_loss": val_pcc_loss,
                        "epoch": epoch_num,
                    }

                    ckpt = {
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch_num,
                        "val_loss": val_loss,
                        "config": {
                            "g_enc": args.g_enc,
                            "e_enc": args.e_enc,
                            "ld_enc": args.ld_enc,
                            "gxe_enc": args.gxe_enc,
                            "block_size": config.block_size,
                            "g_layers": args.g_layers,
                            "ld_layers": args.ld_layers,
                            "mlp_layers": args.mlp_layers,
                            "gxe_layers": args.gxe_layers,
                            "n_head": args.heads,
                            "n_embd": args.emb_size,
                            "loss": args.loss,
                            "alpha": args.alpha
                        },
                        "run": {"id": run.id if 'run' in locals() else None,
                                "name": wandb_run_name}
                    }
                    ckpt_path = Path("checkpoints") / wandb_run_name / f"{val_year}" / f"checkpoint_{epoch_num:04d}.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(ckpt, ckpt_path)
                    print(f"*** validation loss improved: {best_val_loss:.4e} ***")
                else:
                    last_improved += 1
                    print(f"Validation has not improved in {last_improved} steps")
            
            # create a stop flag so all ranks stop training
            stop_flag = torch.tensor([0], device=device)
            if is_main(rank):
                if last_improved > early_stop:
                    print(f"*** no improvement for {early_stop} steps, stopping ***")
                    stop_flag[0] = 1
            # broadcast stop flag to all ranks
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break
                    
            if is_main(rank):
                elapsed = (time.time() - t0) / 60
                print(f"[Epoch {epoch_num}] train={train_loss:.4e} | "
                    f"val={val_loss:.4e} | best_val={best_val_loss:.4e} | "
                    f"elapsed={elapsed:.2f}m")
                
        # end of fold: record best snapshot metrics for this fold
        if is_main(rank) and best_fold_snapshot is not None:
            fold_records.append(best_fold_snapshot)
            wandb.log({
                f"fold/{val_year}/best_val_loss": best_fold_snapshot["best_val_loss"],
                f"fold/{val_year}/best_val_mse": best_fold_snapshot["best_val_mse"],
                f"fold/{val_year}/best_val_pcc_loss": best_fold_snapshot["best_val_pcc_loss"],
            })
    
    # after all folds, log CV summary
    if is_main(rank) and len(fold_records) > 0:
        import numpy as np
        table = wandb.Table(columns=[
            "fold", "train_le", "val_y", "best_val_loss", "best_val_mse", "best_val_pcc_loss"
        ])
        for r in fold_records:
            table.add_data(r["fold"], r["train_le"], r["val_y"], r["best_val_loss"], r["best_val_mse"], r["best_val_pcc_loss"])
        mean_loss = float(np.mean([r["best_val_loss"] for r in fold_records]))
        mean_mse  = float(np.mean([r["best_val_mse"]  for r in fold_records]))
        mean_pcc  = float(np.mean([r["best_val_pcc_loss"]  for r in fold_records]))
        std_mse   = float(np.std( [r["best_val_mse"]  for r in fold_records]))
        std_pcc   = float(np.std( [r["best_val_pcc_loss"]  for r in fold_records]))
        wandb.log({
            "cv/folds_table":   table,
            "cv/mean_val_loss": mean_loss,
            "cv/mean_val_mse":  mean_mse,
            "cv/mean_val_pcc_loss":  mean_pcc,
            "cv/std_val_mse":   std_mse,
            "cv/std_val_pcc_loss":   std_pcc,
        })
        run.summary["cv/mean_val_loss"] = mean_loss
        run.summary["cv/mean_val_mse"]  = mean_mse
        run.summary["cv/mean_val_pcc_loss"]  = mean_pcc
        run.summary["cv/std_val_mse"]   = std_mse
        run.summary["cv/std_val_pcc_loss"]   = std_pcc

    dist.barrier()
    if is_main(rank):
        run.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()