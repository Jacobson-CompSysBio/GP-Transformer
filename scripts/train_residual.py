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
from utils.loss import build_loss, GlobalPearsonCorrLoss, GenomicContrastiveLoss, compute_ibs_similarity, compute_grm_similarity
from utils.utils import *
from utils.utils import EnvStratifiedSampler, str2bool

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

    def _get_arg_or_env(attr, env_key, default, cast=None):
        val = getattr(args, attr, None)
        if val is None:
            env_val = os.getenv(env_key)
            if env_val is None or env_val == "":
                return default
            return cast(env_val) if cast is not None else env_val
        return val

    # reproducibility 
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    # Check if using LEO (Leave-Environment-Out) validation
    leo_val = _get_arg_or_env("leo_val", "LEO_VAL", False, str2bool)
    leo_val_fraction = _get_arg_or_env("leo_val_fraction", "LEO_VAL_FRACTION", 0.15, float)
    
    if is_main(rank) and leo_val:
        print(f"[INFO] Using LEO (Leave-Environment-Out) validation")
        print(f"[INFO] Holding out {leo_val_fraction*100:.0f}% of environments for validation")

    # data (samplers are needed for DDP)
    train_ds = GxE_Dataset(
        split="train",
        data_path='data/maize_data_2014-2023_vs_2024_v2/',
        residual=args.residual,
        scaler=None,
        y_scalers=None,
        scale_targets=args.scale_targets,
        leo_val=leo_val,
        leo_val_fraction=leo_val_fraction,
        leo_seed=args.seed,
    )
    env_scaler = train_ds.scaler
    y_scalers = train_ds.label_scalers
    leo_val_envs = train_ds.leo_val_envs  # Pass to val_ds for consistency
    
    if is_main(rank) and leo_val:
        print(f"[INFO] Train samples: {len(train_ds):,}, Train envs: {train_ds.env_id_tensor.unique().numel()}")
        print(f"[INFO] LEO val envs: {len(leo_val_envs) if leo_val_envs else 0}")
    
    val_ds = GxE_Dataset(
        split="val",
        data_path="data/maize_data_2014-2023_vs_2024_v2/",
        residual=args.residual,
        scaler=env_scaler,
        y_scalers=y_scalers,
        scale_targets=args.scale_targets,
        leo_val=leo_val,
        leo_val_envs=leo_val_envs,  # Use same held-out envs computed by train
    )
    
    if is_main(rank) and leo_val:
        print(f"[INFO] Val samples: {len(val_ds):,}, Val envs: {val_ds.env_id_tensor.unique().numel()}")

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    # Check if using env-stratified sampling (recommended for envpcc loss)
    env_stratified = _get_arg_or_env("env_stratified", "ENV_STRATIFIED", False, str2bool)
    min_samples_per_env = _get_arg_or_env("min_samples_per_env", "MIN_SAMPLES_PER_ENV", 32, int)
    use_batch_sampler = False
    
    if env_stratified and "envpcc" in args.loss.lower():
        if is_main(rank):
            print(f"[INFO] Using environment-stratified sampling for envpcc loss")
            print(f"[INFO] min_samples_per_env = {min_samples_per_env}")
        train_sampler = EnvStratifiedSampler(
            env_ids=train_ds.env_id_tensor.tolist(),
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            min_samples_per_env=min_samples_per_env,
        )
        use_batch_sampler = True
        
        # Debug: show sampler stats (without iterating through all batches)
        if is_main(rank):
            print(f"[DEBUG] Sampler: {len(train_sampler)} batches for this rank, ~{train_sampler.total_chunks} total chunks")
    
    if use_batch_sampler:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            pin_memory=True,
            num_workers=0,  # Disabled: causes slowdown on Lustre
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=0,
        )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        sampler=val_sampler,
        pin_memory=True,
        num_workers=0,
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
    g_encoder_type = _get_arg_or_env("g_encoder_type", "G_ENCODER_TYPE", "dense", str)
    moe_num_experts = _get_arg_or_env("moe_num_experts", "MOE_NUM_EXPERTS", 4, int)
    moe_top_k = _get_arg_or_env("moe_top_k", "MOE_TOP_K", 2, int)
    moe_expert_hidden_dim = _get_arg_or_env("moe_expert_hidden_dim", "MOE_EXPERT_HIDDEN_DIM", None, int)
    moe_shared_expert = _get_arg_or_env("moe_shared_expert", "MOE_SHARED_EXPERT", False, str2bool)
    moe_shared_expert_hidden_dim = _get_arg_or_env("moe_shared_expert_hidden_dim", "MOE_SHARED_EXPERT_HIDDEN_DIM", None, int)
    moe_loss_weight = _get_arg_or_env("moe_loss_weight", "MOE_LOSS_WEIGHT", 0.01, float)
    full_tf_mlp_type = _get_arg_or_env("full_tf_mlp_type", "FULL_TF_MLP_TYPE", None, str)
    if full_tf_mlp_type is None:
        full_tf_mlp_type = g_encoder_type
    if isinstance(full_tf_mlp_type, str):
        full_tf_mlp_type = full_tf_mlp_type.lower()
    else:
        full_tf_mlp_type = "moe" if full_tf_mlp_type else "dense"
    moe_encoder_enabled = (
        (args.full_transformer and full_tf_mlp_type == "moe")
        or (bool(args.g_enc) and str(g_encoder_type).lower() == "moe")
    )

    if args.full_transformer:
        if args.residual:
            model = FullTransformerResidual(
                config,
                mlp_type=full_tf_mlp_type,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                moe_expert_hidden_dim=moe_expert_hidden_dim,
                moe_shared_expert=moe_shared_expert,
                moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
                moe_loss_weight=moe_loss_weight,
                residual=args.residual,
            ).to(device)
            model.detach_ymean_in_sum = args.detach_ymean
        else:
            model = FullTransformer(
                config,
                mlp_type=full_tf_mlp_type,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                moe_expert_hidden_dim=moe_expert_hidden_dim,
                moe_shared_expert=moe_shared_expert,
                moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
                moe_loss_weight=moe_loss_weight,
            ).to(device)
    elif args.residual:
        model = GxE_ResidualTransformer(g_enc=args.g_enc,
                                e_enc=args.e_enc,
                                ld_enc=args.ld_enc,
                                gxe_enc=args.gxe_enc,
                                moe=args.wg,
                                g_encoder_type=g_encoder_type,
                                moe_num_experts=moe_num_experts,
                                moe_top_k=moe_top_k,
                                moe_expert_hidden_dim=moe_expert_hidden_dim,
                                moe_shared_expert=moe_shared_expert,
                                moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
                                moe_loss_weight=moe_loss_weight,
                                residual=args.residual,
                                config=config).to(device)
        model.detach_ymean_in_sum = args.detach_ymean
    else:
        model = GxE_Transformer(g_enc=args.g_enc,
                                e_enc=args.e_enc,
                                ld_enc=args.ld_enc,
                                gxe_enc=args.gxe_enc,
                                moe=args.wg,
                                g_encoder_type=g_encoder_type,
                                moe_num_experts=moe_num_experts,
                                moe_top_k=moe_top_k,
                                moe_expert_hidden_dim=moe_expert_hidden_dim,
                                moe_shared_expert=moe_shared_expert,
                                moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
                                moe_loss_weight=moe_loss_weight,
                                config=config).to(device)
    if is_main(rank):
        model.print_trainable_parameters()
    # Residual models have multiple output heads (ymean_head, head) and auxiliary losses
    # that may not flow gradients through all parameters in every forward pass.
    # MoE models have expert routing that may leave some experts unused.
    # In both cases, we need find_unused_parameters=True to avoid DDP sync errors.
    find_unused = bool(args.wg) or moe_encoder_enabled or bool(args.residual) or bool(getattr(args, 'contrastive_loss', False))
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = build_loss(args.loss, args.loss_weights)

    # optional contrastive loss for genomic embeddings
    use_contrastive = getattr(args, 'contrastive_loss', False)
    if use_contrastive:
        contrastive_weight = getattr(args, 'contrastive_weight', 0.1)
        contrastive_temperature = getattr(args, 'contrastive_temperature', 0.1)
        contrastive_sim_type = getattr(args, 'contrastive_sim_type', 'grm')
        contrastive_loss_type = getattr(args, 'contrastive_loss_type', 'mse')
        contrastive_loss_fn = GenomicContrastiveLoss(
            temperature=contrastive_temperature,
            similarity_type=contrastive_sim_type,
            loss_type=contrastive_loss_type,
        )
        if is_main(rank):
            print(f"[INFO] Using genomic contrastive loss:")
            print(f"       weight={contrastive_weight}, temperature={contrastive_temperature}")
            print(f"       similarity_type={contrastive_sim_type}, loss_type={contrastive_loss_type}")
    else:
        contrastive_loss_fn = None
        contrastive_weight = 0.0

    # other options
    batches_per_epoch = len(train_loader)
    batches_per_eval = len(val_loader)
    # Use effective training horizon (early_stop), not max_epochs, for LR schedule
    # Otherwise cosine decay never activates before early stopping kicks in
    effective_epochs = min(args.early_stop * 2, args.num_epochs)  # ~2x early_stop as budget
    total_iters = effective_epochs * batches_per_epoch
    warmup_iters = batches_per_epoch * 5  # warmup for ~5 epochs (large batch needs longer warmup)
    lr_decay_iters = total_iters  # cosine spans entire effective window
    max_lr, min_lr = (args.lr), (0.1 * args.lr)  # 10x decay ratio
    max_epochs = args.num_epochs
    eval_interval = batches_per_epoch
    early_stop = args.early_stop

    if is_main(rank):
        print(f"[INFO] batches_per_epoch = {batches_per_epoch}, batch_size = {args.batch_size}")
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
                             "loss_weights": args.loss_weights,
                             "residual": args.residual,
                             "detach_ymean": args.detach_ymean,
                             "lambda_ymean": args.lambda_ymean,
                             "lambda_resid": args.lambda_resid,
                             "n_embd": args.emb_size,
                             "gxe_layers": args.gxe_layers,
                             "g_layers": args.g_layers,
                             "ld_layers": args.ld_layers,
                             "mlp_layers": args.mlp_layers,
                             "heads": args.heads,
                             "dropout": args.dropout,
                             "lr": args.lr,
                             "weight_decay": args.weight_decay,
                             "batch_size": args.batch_size,
                             "gbs": args.gbs,
                             "early_stop": args.early_stop,
                             "g_encoder_type": g_encoder_type,
                             "moe_num_experts": moe_num_experts,
                             "moe_top_k": moe_top_k,
                             "moe_expert_hidden_dim": moe_expert_hidden_dim,
                             "moe_shared_expert": moe_shared_expert,
                             "moe_shared_expert_hidden_dim": moe_shared_expert_hidden_dim,
                             "moe_loss_weight": moe_loss_weight,
                             "full_tf_mlp_type": full_tf_mlp_type,
                             "full_transformer": args.full_transformer,
                             "contrastive_loss": use_contrastive,
                             "contrastive_weight": contrastive_weight,},
                             allow_val_change=True)
        for name in loss_function.names:
            run.define_metric(f"train_loss/{name}", step_metric="iter_num")
            run.define_metric(f"train_loss_epoch/{name}", step_metric="epoch")
            run.define_metric(f"val_loss/{name}", step_metric="epoch")
        
        # track contrastive loss
        if use_contrastive:
            run.define_metric("train_loss/contrastive", step_metric="iter_num")
            run.define_metric("train_loss/contrastive_weight_eff", step_metric="iter_num")

        # track residual losses
        if args.residual:
            run.define_metric("aux_ymean_loss", step_metric="iter_num")
            run.define_metric("aux_resid_loss", step_metric="iter_num")
        if moe_encoder_enabled:
            run.define_metric("train_loss/moe_lb", step_metric="iter_num")
            run.define_metric("train_loss_epoch/moe_lb", step_metric="epoch")
            run.define_metric("val_loss/moe_lb", step_metric="epoch")
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
        
        # Timing diagnostics for first epoch
        if epoch_num == 0 and is_main(rank):
            t_epoch_start = time.time()

        pbar = tqdm(total=eval_interval,
                    desc=f"Rank {rank} Train",
                    disable=not is_main(rank))
        
        # training steps
        step_times = []
        for step_idx, (xb, yb) in enumerate(train_loader):
            if epoch_num == 0 and is_main(rank):
                t_step_start = time.time()
            
            for k, v in xb.items():
                xb[k] = v.to(device, non_blocking=True)
            yb = _move_to_device(yb, device) 

            # fwd/bwd pass
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if use_contrastive:
                    logits, g_embeddings = model(xb, return_g_embeddings=True)
                else:
                    logits = model(xb)
                    g_embeddings = None

                # residual: compute main and aux losses
                loss_aux_ymean = None
                loss_aux_resid = None
                if args.residual:
                    pred_total = logits["total"]
                    loss_main, loss_parts = loss_function(pred_total, yb["total"], env_id=yb["env_id"])
                    loss_aux_ymean = F.mse_loss(logits["ymean"], yb["ymean"])
                    loss_aux_resid = F.mse_loss(logits["resid"], yb["resid"])
                    loss = loss_main + (args.lambda_ymean * loss_aux_ymean) + (args.lambda_resid * loss_aux_resid)
                else:
                    loss, loss_parts = loss_function(logits, yb["y"], env_id=yb["env_id"])

                # Add contrastive loss if enabled (with warmup)
                contrastive_warmup_epochs = 50
                if use_contrastive and g_embeddings is not None and epoch_num >= contrastive_warmup_epochs:
                    contrastive_loss = contrastive_loss_fn(g_embeddings, g_data=xb["g_data"])
                    warmup_factor = min(1.0, (epoch_num - contrastive_warmup_epochs) / 50.0)
                    effective_weight = contrastive_weight * warmup_factor
                    loss = loss + effective_weight * contrastive_loss
                    loss_parts["contrastive"] = float(contrastive_loss.detach().item())
                    loss_parts["contrastive_weight_eff"] = effective_weight
                elif use_contrastive:
                    loss_parts["contrastive"] = 0.0
                    loss_parts["contrastive_weight_eff"] = 0.0

                moe_aux_loss = getattr(model.module, "moe_aux_loss", None)
                if moe_aux_loss is not None:
                    # Don't detach - gradients need to flow to gate network for load balancing
                    # DDP handles this with find_unused_parameters=True
                    loss = loss + moe_aux_loss
                    loss_parts["moe_lb"] = float(moe_aux_loss.detach().item())

            if torch.isnan(loss):
                raise RuntimeError("Loss is NaN, stopping training.")

            # compute losses for wandb logging
            aux_ymean_val = loss_aux_ymean.detach() if loss_aux_ymean is not None else None
            aux_resid_val = loss_aux_resid.detach() if loss_aux_resid is not None else None

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
                for k, v in loss_parts.items():
                    log_payload[f"train_loss/{k}"] = float(v)
                if aux_ymean_val is not None:
                    log_payload["aux_ymean_loss"] = float(aux_ymean_val.item())
                if aux_resid_val is not None:
                    log_payload["aux_resid_loss"] = float(aux_resid_val.item())
                wandb.log(log_payload)
            
            # Timing diagnostics
            if epoch_num == 0 and is_main(rank):
                step_times.append(time.time() - t_step_start)
                if step_idx < 5 or step_idx % 20 == 0:
                    print(f"[TIMING] Step {step_idx}: {step_times[-1]:.3f}s (avg: {sum(step_times)/len(step_times):.3f}s)")
            
            iter_num += 1
            pbar.update(1)
        pbar.close()
        
        # End of epoch timing
        if epoch_num == 0 and is_main(rank):
            t_epoch_end = time.time()
            print(f"[TIMING] Epoch 0: {t_epoch_end - t_epoch_start:.1f}s total, {len(step_times)} steps, {sum(step_times)/len(step_times):.3f}s/step avg")

        ### evaluation ###
        with torch.no_grad():
            model.eval()

            def _gather_predictions(loader, dataset_len):
                """Gather ALL predictions/targets across batches and DDP ranks,
                then compute metrics on the full dataset.  This aligns the
                validation metric with how eval.py computes test metrics."""
                all_preds = []
                all_targets = []
                all_env_ids = []
                all_ymean_preds = []
                all_ymean_targets = []
                all_resid_preds = []
                all_resid_targets = []

                for xb, yb in loader:
                    xb = _move_to_device(xb, device)
                    yb = _move_to_device(yb, device)

                    out = model(xb)
                    if args.residual:
                        all_preds.append(out["total"].squeeze(-1))
                        all_targets.append(yb["total"].squeeze(-1))
                        all_ymean_preds.append(out["ymean"].squeeze(-1))
                        all_ymean_targets.append(yb["ymean"].squeeze(-1))
                        all_resid_preds.append(out["resid"].squeeze(-1))
                        all_resid_targets.append(yb["resid"].squeeze(-1))
                    else:
                        all_preds.append(out.squeeze(-1))
                        all_targets.append(yb["y"].squeeze(-1))
                    all_env_ids.append(yb["env_id"])

                # Concatenate local predictions
                local_preds = torch.cat(all_preds) if all_preds else torch.empty(0, device=device)
                local_targets = torch.cat(all_targets) if all_targets else torch.empty(0, device=device)
                local_env_ids = torch.cat(all_env_ids) if all_env_ids else torch.empty(0, dtype=torch.long, device=device)
                local_ymean_p = torch.cat(all_ymean_preds) if all_ymean_preds else None
                local_ymean_t = torch.cat(all_ymean_targets) if all_ymean_targets else None
                local_resid_p = torch.cat(all_resid_preds) if all_resid_preds else None
                local_resid_t = torch.cat(all_resid_targets) if all_resid_targets else None

                # All-gather across DDP ranks to reconstruct full dataset
                # DistributedSampler may pad, so gather all then trim to dataset_len
                def _all_gather_flat(t):
                    local_n = torch.tensor([t.shape[0]], device=device)
                    all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
                    dist.all_gather(all_n, local_n)
                    max_n = max(x.item() for x in all_n)
                    # Pad to max_n
                    padded = torch.zeros(max_n, device=device, dtype=t.dtype)
                    padded[:t.shape[0]] = t
                    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                    dist.all_gather(gathered, padded)
                    # Trim each rank's contribution to actual size
                    parts = [gathered[i][:all_n[i].item()] for i in range(world_size)]
                    return torch.cat(parts)

                full_preds = _all_gather_flat(local_preds)[:dataset_len]
                full_targets = _all_gather_flat(local_targets)[:dataset_len]
                full_env_ids = _all_gather_flat(local_env_ids.float()).long()[:dataset_len]

                full_ymean_p = _all_gather_flat(local_ymean_p)[:dataset_len] if local_ymean_p is not None else None
                full_ymean_t = _all_gather_flat(local_ymean_t)[:dataset_len] if local_ymean_t is not None else None
                full_resid_p = _all_gather_flat(local_resid_p)[:dataset_len] if local_resid_p is not None else None
                full_resid_t = _all_gather_flat(local_resid_t)[:dataset_len] if local_resid_t is not None else None

                return (full_preds, full_targets, full_env_ids,
                        full_ymean_p, full_ymean_t, full_resid_p, full_resid_t)

            def eval_loader(loader, dataset_len):
                """Compute validation metrics using gather-then-compute approach.
                This computes per-env Pearson r on the FULL gathered dataset,
                matching the methodology used in eval.py for test metrics."""
                (full_preds, full_targets, full_env_ids,
                 full_ymean_p, full_ymean_t, full_resid_p, full_resid_t) = \
                    _gather_predictions(loader, dataset_len)

                # Compute the main loss on the FULL dataset (same metric as test)
                ltot, lparts = loss_function(full_preds, full_targets, env_id=full_env_ids)

                # Aux residual metrics
                mean_aux_ymean = 0.0
                mean_aux_resid = 0.0
                if full_ymean_p is not None and full_ymean_t is not None:
                    mean_aux_ymean = F.mse_loss(full_ymean_p, full_ymean_t).item()
                if full_resid_p is not None and full_resid_t is not None:
                    mean_aux_resid = F.mse_loss(full_resid_p, full_resid_t).item()

                mean_parts = {k: float(v) for k, v in lparts.items()}
                return ltot, mean_parts, mean_aux_ymean, mean_aux_resid

            train_total, train_parts, aux_train_ymean, aux_train_resid = eval_loader(train_loader, len(train_ds))
            val_total, val_parts, aux_val_ymean, aux_val_resid = eval_loader(val_loader, len(val_ds))

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
            if args.residual:
                log_epoch_payload["aux_ymean_mse_epoch"] = aux_val_ymean
                log_epoch_payload["aux_resid_mse_epoch"] = aux_val_resid
            wandb.log(log_epoch_payload)

            if val_total < best_val_loss:
                best_val_loss, last_improved = val_total, 0
                
                # collect env scaler and y scalers
                env_scaler_payload = {
                    "mean": env_scaler.mean_.tolist(),
                    "scale": env_scaler.scale_.tolist(),
                    "var": env_scaler.var_.tolist(),
                    "n_features_in": int(train_ds.scaler.n_features_in_),
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
                    "val_loss": float(val_total.item()),
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
                        "g_encoder_type": g_encoder_type,
                        "moe_num_experts": moe_num_experts,
                        "moe_top_k": moe_top_k,
                        "moe_expert_hidden_dim": moe_expert_hidden_dim,
                        "moe_shared_expert": moe_shared_expert,
                        "moe_shared_expert_hidden_dim": moe_shared_expert_hidden_dim,
                        "moe_loss_weight": moe_loss_weight,
                        "full_tf_mlp_type": full_tf_mlp_type,
                        "loss": args.loss,
                        "loss_weights": args.loss_weights,
                        "residual": args.residual,
                        "lambda_ymean": args.lambda_ymean,
                        "lambda_resid": args.lambda_resid,
                        "detach_ymean": args.detach_ymean,
                        "full_transformer": args.full_transformer,
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
            print(f"[Epoch {epoch_num}] train={train_total.item():.4e} | "
                  f"val={val_total.item():.4e} | best_val={best_val_loss:.4e} | "
                  f"elapsed={elapsed:.2f}m")

    dist.barrier()
    if is_main(rank):
        run.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()
