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
from utils.utils import *

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
    wandb_run_name = make_run_name(args)

    device, local_rank, rank, world_size = setup_ddp()

    # reproducibility 
    set_seed(args.seed + rank)  # different seed for each rank
    g = torch.Generator()
    g.manual_seed(args.seed + rank)

    # data (samplers are needed for DDP)
    train_ds = GxE_Dataset(
        split="train",
        data_path='data/maize_data_2014-2023_vs_2024_v2/',
        scaler=None,
        y_scalers=None, # train will fit the scalers
        scale_targets=args.scale_targets
    )
    env_scaler = train_ds.scaler
    y_scalers = train_ds.label_scalers
    val_ds = GxE_Dataset(
        split="val",
        data_path="data/maize_data_2014-2023_vs_2024_v2/",
        scaler=env_scaler,
        y_scalers=y_scalers,
        scale_targets=args.scale_targets
    )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    # set up config
    config = Config(block_size=train_ds.block_size,
                    n_head=args.heads,
                    n_g_layer=args.g_layers,
                    n_ld_layer=args.ld_layers,
                    n_mlp_layer=args.mlp_layers,
                    n_gxe_layer=args.gxe_layers,
                    n_embd=args.emb_size,
                    dropout=args.dropout,
                    n_env_fts=train_ds.n_env_fts)
    model = GxE_Transformer(g_enc=args.g_enc,
                            e_enc=args.e_enc,
                            ld_enc=args.ld_enc,
                            gxe_enc=args.gxe_enc,
                            moe=args.moe,
                            config=config).to(device)
    if is_main(rank):
        model.print_trainable_parameters()
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # build loss
    loss_function = build_loss(args.loss, args.loss_weights)

    # other options
    batches_per_epoch = len(train_loader)
    total_iters = args.num_epochs * batches_per_epoch
    warmup_iters = batches_per_epoch * 1  # warmup for ~1 epoch
    lr_decay_iters = int(total_iters * 0.6)  # decay over ~60% of training
    max_lr, min_lr = (args.lr), (0.1 * args.lr)  # keep a higher floor for stability
    max_epochs = args.num_epochs
    eval_interval = batches_per_epoch
    early_stop = args.early_stop

    if is_main(rank):
        print(f"Training on {batches_per_epoch * args.batch_size * int(os.getenv('SLURM_NNODES'))* 8} samples for {args.num_epochs} epochs.")

    ### wandb logging ###
    if is_main(rank):
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=wandb_run_name
        )

        # make checkpoint dir that matches run name
        run_ckpt_dir = Path("checkpoints") / wandb_run_name
        if run_ckpt_dir.exists():
            shutil.rmtree(run_ckpt_dir)
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # write run id to logs
        run_id_file = os.environ.get("WANDB_RUN_ID_FILE")
        if run_id_file:
            with open(run_id_file, "w") as f:
                f.write(run.id.strip())
            print(f"[INFO] WandB run id written to {run_id_file}: {run.id}")

        cdir_file = os.environ.get("CHECKPOINT_DIR_FILE")
        if cdir_file:
            with open(cdir_file, "w") as f:
                f.write(str(run_ckpt_dir.resolve()))
            print(f"[INFO] Checkpoint dir written to {cdir_file}: {run_ckpt_dir.resolve()}")

        run.define_metric("iter_num")
        run.define_metric("train_loss", step_metric="iter_num")
        run.define_metric("learning_rate", step_metric="iter_num")

        # use epochs as steps for epoch-level metrics
        run.define_metric("epoch")
        run.define_metric("train_loss_epoch", step_metric="epoch")
        run.define_metric("val_loss", step_metric="epoch")

        # loss tracking
        wandb.config.update({"loss": args.loss,
                             "loss_weights": args.loss_weights},
                             allow_val_change=True)
        for name in loss_function.names:
            run.define_metric(f"train_loss/{name}", step_metric="iter_num")
            run.define_metric(f"train_loss_epoch/{name}", step_metric="epoch")
            run.define_metric(f"val_loss/{name}", step_metric="epoch")

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
            y_true = yb["y"].to(device, non_blocking=True).float()
            env_id = yb["env_id"].to(device, non_blocking=True).long()

            # fwd/bwd pass
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = model(xb)
                loss_total, loss_parts = loss_function(preds, y_true, env_id=env_id)
            if torch.isnan(loss_total):
                raise RuntimeError("Loss is NaN, stopping training.")

            # apply learning rate schedule              
            lr = get_lr(iter_num,
                        warmup_iters,
                        lr_decay_iters,
                        max_lr,
                        min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # clip gradients on bwd to avoid unstable training 
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # log wandb
            if is_main(rank):
                log_payload = {
                    "train_loss": loss_total.item(),
                    "learning_rate": lr,
                    "iter_num": iter_num,
                }
                for k, v in loss_parts.items():
                    log_payload[f"train_loss/{k}"] = float(v)
                wandb.log(log_payload)
            iter_num += 1
            pbar.update(1)
        pbar.close()

        ### evaluation ###
        with torch.no_grad():
            model.eval()

            def eval_loader(loader, max_batches=None):
                """
                evaluates loader, returns:
                    mean_total_loss, parts_mean_dict, nbatches
                """

                # accumulate as tensors to allow dist.all_reduce
                total_loss = torch.tensor(0.0, device=device)
                parts_acc = {name: torch.tensor(0.0, device=device) for name in loss_function.names}
                n_batches = torch.tensor(0.0, device=device) 

                for i, (xb, yb) in enumerate(loader):
                    # if we hit eval_iters limit, break
                    if (max_batches is not None) and (i >= max_batches):
                        break
                    
                    # send to device
                    xb = {k: v.to(device, non_blocking=True) for k, v in xb.items()}
                    y_true = yb["y"].to(device, non_blocking=True).float()
                    env_id = yb["env_id"].to(device, non_blocking=True).long()

                    # fwd
                    preds = model(xb)
                    ltot, lparts = loss_function(preds, y_true, env_id=env_id)
                    total_loss += ltot
                    for k in parts_acc:
                        parts_acc[k] += torch.tensor(lparts[k], device=device)
                    n_batches += 1.0

                # allreduce w/ sum
                dist.all_reduce(total_loss)
                for t in parts_acc.values():
                    dist.all_reduce(t)
                dist.all_reduce(n_batches)
                
                # convert to global means with zero div guard
                nb = max(1.0, float(n_batches.item()))
                mean_total = total_loss / nb
                mean_parts = {k: (v / nb).item() for k, v in parts_acc.items()}

                return mean_total, mean_parts, nb
            
            # reshuffle train sampler s.t. sampled subset is different each epoch
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(10_000 + epoch_num)
            # eval on subset of train for speed, full val
            train_total, train_parts, _ = eval_loader(train_loader, max_batches=int(math.ceil(len(val_loader) / world_size))) 
            val_total, val_parts, _ = eval_loader(val_loader, max_batches=None)

        # log eval / early stop (only rank 0)
        if is_main(rank):
            log_epoch_payload = {
                "val_loss": val_total.item(),
                "train_loss_epoch": train_total.item(),
                "epoch": epoch_num,
            }
            for k, v in train_parts.items():
                log_epoch_payload[f"train_loss_epoch/{k}"] = float(v)
            for k, v in val_parts.items():
                log_epoch_payload[f"val_loss/{k}"] = float(v)
            wandb.log(log_epoch_payload)

            if val_total < best_val_loss:
                best_val_loss, last_improved = val_total, 0
                # collect env scaler and y scalers
                env_scaler_payload = {
                    "mean": env_scaler.mean_.tolist(),
                    "scale": env_scaler.scale_.tolist(),
                    "var": env_scaler.var_.tolist(),
                    "n_features_in": int(train_ds.scaler.n_features_in_),
                    "feature_names_in": list(train_ds.e_cols)
                }

                label_scalers_payload = None
                if hasattr(train_ds, 'label_scalers') and train_ds.label_scalers:
                    label_scalers_payload = {
                        k: {"mean": float(v.mean), "std": float(v.std)}
                        for k, v in train_ds.label_scalers.items()
                    }

                ckpt = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch_num,
                    "val_loss": val_total,
                    "config": {
                        "g_enc": args.g_enc,
                        "e_enc": args.e_enc,
                        "ld_enc": args.ld_enc,
                        "gxe_enc": args.gxe_enc,
                        "block_size": config.block_size,
                        "n_env_fts": config.n_env_fts,
                        "g_layers": args.g_layers,
                        "ld_layers": args.ld_layers,
                        "mlp_layers": args.mlp_layers,
                        "gxe_layers": args.gxe_layers,
                        "n_head": args.heads,
                        "n_embd": args.emb_size,
                        "loss": args.loss,
                        "loss_weights": args.loss_weights,
                        "scale_targets": args.scale_targets,
                    },
                    "env_scaler": env_scaler_payload,
                    "y_scalers": label_scalers_payload,
                    "run": {"id": run.id if 'run' in locals() else None,
                            "name": wandb_run_name}
                }
                ckpt_path = Path("checkpoints") / wandb_run_name / f"checkpoint_{epoch_num:04d}.pt"
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
            print(f"[Epoch {epoch_num}] train={train_total:.4e} | "
                  f"val={val_total:.4e} | best_val={best_val_loss:.4e} | "
                  f"elapsed={elapsed:.2f}m")

    dist.barrier()
    if is_main(rank):
        run.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()
