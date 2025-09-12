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

# residual: small helper to move nested dicts to device
def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device) 
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj

### main ###
def main():
    # setup
    args = parse_args()
    wandb_run_name = make_run_name(args)

    device, local_rank, rank, world_size = setup_ddp()

    # reproducibility 
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # data (samplers are needed for DDP)
    train_ds = GxE_Dataset(
        split="train",
        data_path='data/maize_data_2014-2023_vs_2024/',
        index_map_path='data/maize_data_2014-2023_vs_2024/location_2014_2023.csv',
        residual=args.residual,
        scaler=None
    )
    scaler = train_ds.scaler
    val_ds = GxE_Dataset(
        split="val",
        data_path="data/maize_data_2014-2023_vs_2024/",
        index_map_path="data/maize_data_2014-2023_vs_2024/location_2014_2023.csv",
        residual=args.residual,
        scaler=scaler
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
    if args.residual:
        model = GxE_ResidualTransformer(g_enc=args.g_enc,
                                e_enc=args.e_enc,
                                ld_enc=args.ld_enc,
                                gxe_enc=args.gxe_enc,
                                moe=args.moe,
                                residual=args.residual,
                                config=config).to(device)
        model.detach_ymean_in_sum = args.detach_ymean
    else:
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
    batches_per_eval = len(val_loader)
    total_iters = args.num_epochs * batches_per_epoch
    warmup_iters = batches_per_epoch  # warmup for 1 epoch - should be enough
    lr_decay_iters = total_iters * .5 
    max_lr, min_lr = (args.lr), (0.001 * args.lr) 
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
                             "alpha": args.alpha,
                             "residual": args.residual,
                             "detach_ymean": args.detach_ymean,
                             "lambda_ymean": args.lambda_ymean,
                             "lambda_resid": args.lambda_resid},
                             allow_val_change=True)
        if args.loss == "both":
            # batch level            
            run.define_metric("mse", step_metric="iter_num")
            run.define_metric("pcc_loss", step_metric="iter_num")

            # epoch level
            run.define_metric("mse_epoch", step_metric="epoch")
            run.define_metric("pcc_loss_epoch", step_metric="epoch")
            run.define_metric("mse_epoch", step_metric="epoch")
            run.define_metric("pcc_loss_epoch", step_metric="epoch")
        
        # track residual losses 
        if args.residual:
            run.define_metric("aux_ymean_loss", step_metric="iter_num")
            run.define_metric("aux_resid_loss", step_metric="iter_num")
            run.define_metric("aux_ymean_mse_epoch", step_metric="epoch")
            run.define_metric("aux_resid_mse_epoch", step_metric="epoch")

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
            yb = _move_to_device(yb, device) 

            # fwd/bwd pass
            logits = model(xb)

            # residual: compute main and aux losses
            if args.residual:
                pred_total = logits['total']
                loss_main = loss_function(pred_total, yb['total'])
                loss_aux_ymean = F.mse_loss(logits['ymean'], yb['ymean'])
                loss_aux_resid = F.mse_loss(logits['resid'], yb['resid'])
                loss = (loss_main + (args.lambda_ymean * loss_aux_ymean) + (args.lambda_resid * loss_aux_resid))
            
            # non-residual: just compute main loss
            else:
                loss = loss_function(logits, yb)

            if torch.isnan(loss):
                raise RuntimeError("Loss is NaN, stopping training.")

            # compute losses for wandb logging 
            train_mse_val = None
            train_pcc_loss_val = None
            aux_ymean_val = None
            aux_resid_val = None

            with torch.no_grad():
                if args.residual:
                    # log on total
                    pred_total = logits['total']
                    tgt_total = yb['total']
                    if args.loss == "both":
                        train_mse_val = mse_loss_log(pred_total, tgt_total)
                        train_pcc_loss_val = pcc_loss_log(pred_total, tgt_total)
                    # aux logs regardless of main loss choice
                    if "ymean" in logits:
                        aux_ymean_val = F.mse_loss(logits['ymean'], yb['ymean'])
                    if "resid" in logits:
                        aux_resid_val = F.mse_loss(logits['resid'], yb['resid'])
                else:
                    if args.loss == "both":
                        with torch.no_grad():
                            train_mse_val = mse_loss_log(logits, yb)
                            train_pcc_loss_val = pcc_loss_log(logits, yb)

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
                log_payload = {
                    "train_loss": loss.item(),
                    "learning_rate": lr,
                    "iter_num": iter_num,
                }
                if args.loss == "both" and train_mse_val is not None:
                    log_payload["mse"] = float(train_mse_val.item())
                    log_payload["pcc_loss"] = float(train_pcc_loss_val.item())
                if aux_ymean_val is not None:
                    log_payload["aux_ymean_loss"] = float(aux_ymean_val.item())
                if aux_resid_val is not None:
                    log_payload["aux_resid_loss"] = float(aux_resid_val.item())
                wandb.log(log_payload)
            iter_num += 1
            pbar.update(1)
        pbar.close()

        ### evaluation ###
        with torch.no_grad():
            model.eval()
            train_loss_accum = 0
            val_loss_accum = 0

            # for wandb logging
            train_mse_accum = torch.tensor(0.0, device=device)
            train_pcc_loss_accum = torch.tensor(0.0, device=device)
            val_mse_accum = torch.tensor(0.0, device=device)
            val_pcc_loss_accum = torch.tensor(0.0, device=device)

            aux_train_ymean_accum = torch.tensor(0.0, device=device)
            aux_train_resid_accum = torch.tensor(0.0, device=device)
            aux_val_ymean_accum = torch.tensor(0.0, device=device)
            aux_val_resid_accum = torch.tensor(0.0, device=device)

            for (xbt, ybt), (xbv, ybv) in zip(train_loader, val_loader):
                for k, v in xbt.items():
                    xbt[k] = v.to(device, non_blocking=True)
                ybt = _move_to_device(ybt, device) 
                for k, v in xbv.items():
                    xbv[k] = v.to(device, non_blocking=True)
                ybv = _move_to_device(ybv, device)

                out_train = model(xbt)
                out_val = model(xbv)

                if args.residual:
                    loss_train = loss_function(out_train['total'], ybt['total'])
                    loss_val = loss_function(out_val['total'], ybv['total'])
                    train_loss_accum += loss_train
                    val_loss_accum += loss_val

                    # aux accumulation (for monitoring)
                    aux_train_ymean_accum += F.mse_loss(out_train['ymean'], ybt['ymean'])
                    aux_train_resid_accum += F.mse_loss(out_train['resid'], ybt['resid'])
                    aux_val_ymean_accum += F.mse_loss(out_val['ymean'], ybv['ymean'])
                    aux_val_resid_accum += F.mse_loss(out_val['resid'], ybv['resid'])

                    if args.loss == "both":
                        train_mse_accum += mse_loss_log(out_train['total'], ybt['total'])
                        train_pcc_loss_accum += pcc_loss_log(out_train['total'], ybt['total'])
                        val_mse_accum += mse_loss_log(out_val['total'], ybv['total'])
                        val_pcc_loss_accum += pcc_loss_log(out_val['total'], ybv['total'])
                else:
                    loss_train = loss_function(out_train, ybt)
                    loss_val = loss_function(out_val, ybv)
                    train_loss_accum += loss_train
                    val_loss_accum += loss_val

                    if args.loss == "both":
                        train_mse_accum += mse_loss_log(out_train, ybt)
                        train_pcc_loss_accum += pcc_loss_log(out_train, ybt)
                        val_mse_accum += mse_loss_log(out_val, ybv)
                        val_pcc_loss_accum += pcc_loss_log(out_val, ybv)
                
                if train_loss_accum.isnan() or val_loss_accum.isnan():
                    raise RuntimeError("Loss is NaN during evaluation, stopping training.")
        
        # aggregate losses over all ranks
        for t in (train_loss_accum, val_loss_accum,
                  train_mse_accum, train_pcc_loss_accum,
                  val_mse_accum, val_pcc_loss_accum,
                  aux_train_ymean_accum, aux_train_resid_accum,
                  aux_val_ymean_accum, aux_val_resid_accum): 
            dist.all_reduce(t)

        train_loss = (train_loss_accum / batches_per_eval / world_size).item()
        val_loss = (val_loss_accum / batches_per_eval / world_size).item()
        if args.loss == "both":
            train_mse = (train_mse_accum / batches_per_eval / world_size).item()
            train_pcc_loss = (train_pcc_loss_accum / batches_per_eval / world_size).item()
            val_mse = (val_mse_accum / batches_per_eval / world_size).item()
            val_pcc_loss = (val_pcc_loss_accum / batches_per_eval / world_size).item()

        aux_train_ymean = (aux_train_ymean_accum / batches_per_eval / world_size).item()
        aux_train_resid = (aux_train_resid_accum / batches_per_eval / world_size).item()
        aux_val_ymean = (aux_val_ymean_accum / batches_per_eval / world_size).item()
        aux_val_resid = (aux_val_resid_accum / batches_per_eval / world_size).item()

        # log eval / early stop (only rank 0)
        if is_main(rank):
            log_epoch_payload = {
                "val_loss": val_loss,
                "train_loss_epoch": train_loss,
                "epoch": epoch_num,
                "aux_ymean_mse_epoch": aux_val_ymean,
                "aux_resid_mse_epoch": aux_val_resid
            }
            if args.loss == "both":
                log_epoch_payload.update({
                    "train_mse": train_mse,
                    "train_pcc_loss": train_pcc_loss,
                    "val_mse": val_mse,
                    "val_pcc_loss": val_pcc_loss,
                })
            wandb.log(log_epoch_payload)

            if val_loss < best_val_loss:
                best_val_loss, last_improved = val_loss, 0
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
                        "alpha": args.alpha,
                        "residual": args.residual,
                        "lambda_ymean": args.lambda_ymean,
                        "lambda_resid": args.lambda_resid,
                        "detach_ymean": args.detach_ymean,
                    },
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
            print(f"[Epoch {epoch_num}] train={train_loss:.4e} | "
                  f"val={val_loss:.4e} | best_val={best_val_loss:.4e} | "
                  f"elapsed={elapsed:.2f}m")

    dist.barrier()
    if is_main(rank):
        run.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()