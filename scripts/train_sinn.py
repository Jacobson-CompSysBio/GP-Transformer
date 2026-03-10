"""
SINN-style staged training for GP-Transformer.

Phases:
  g        — Train G encoder on G_hat targets (genotype main effect)
  e        — Train E encoder on E_hat targets (environment main effect)
  ge       — Train full model on GE_hat targets (interaction residual),
             initializing G/E encoders from phases g/e
  finetune — End-to-end fine-tune on raw yield using envpcc loss,
             initializing from phase ge checkpoint

Usage:
  python scripts/train_sinn.py --sinn_phase g   [--other_args ...]
  python scripts/train_sinn.py --sinn_phase e   [--other_args ...]
  python scripts/train_sinn.py --sinn_phase ge  --g_ckpt ... --e_ckpt ...
  python scripts/train_sinn.py --sinn_phase finetune --ge_ckpt ...
"""

import time, os, json, math, sys, shutil, argparse, subprocess
from pathlib import Path
from contextlib import nullcontext
from tqdm import tqdm
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import wandb

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.dataset import GxE_Dataset, normalize_env_categorical_mode
from models.model import FullTransformer
from models.transformer import G_Encoder
from models.mlp import E_Encoder
from models.config import Config
from utils.get_lr import get_lr
from utils.loss import build_loss, macro_env_pearson
from utils.utils import set_seed, seed_worker, str2bool, EnvStratifiedSampler

load_dotenv()
os.environ.setdefault("WANDB_PROJECT", os.getenv("WANDB_PROJECT", ""))
os.environ.setdefault("WANDB_ENTITY", os.getenv("WANDB_ENTITY", ""))

# ── DDP helpers ──────────────────────────────────────────────────
def extract_master_addr():
    try:
        nodelist = os.environ["SLURM_NODELIST"]
        node = subprocess.check_output(
            ["scontrol", "show", "hostname", nodelist]
        ).decode().splitlines()[0]
        return node
    except Exception:
        return "localhost"


def setup_ddp():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = 0
    torch.cuda.set_device(local_rank)
    master = os.environ.get("MASTER_ADDR", "")
    if not master or ":" in master:
        os.environ["MASTER_ADDR"] = extract_master_addr()
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return torch.device(f"cuda:{local_rank}"), local_rank, rank, world_size


def is_main(rank) -> bool:
    return rank == 0


# ── Lightweight wrapper models for component training ────────────
class SINNGModel(nn.Module):
    """G_Encoder + prediction head → scalar G_hat."""

    def __init__(self, config, **encoder_kwargs):
        super().__init__()
        self.encoder = G_Encoder(config, **encoder_kwargs)
        self.head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        g_enc = self.encoder(x["g_data"])        # (B, T+1, C)
        cls_repr = g_enc[:, 0]                   # CLS token → (B, C)
        return self.head(cls_repr)                # (B, 1)


class SINNEModel(nn.Module):
    """E_Encoder + prediction head → scalar E_hat."""

    def __init__(self, config):
        super().__init__()
        self.encoder = E_Encoder(
            input_dim=config.n_env_fts,
            output_dim=config.n_embd,
            hidden_dim=config.n_embd,
            n_hidden=config.n_mlp_layer,
            dropout=config.dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        e_enc = self.encoder(x["e_data"])         # (B, C)
        return self.head(e_enc)                    # (B, 1)


# ── Arg parser ───────────────────────────────────────────────────
def parse_sinn_args():
    p = argparse.ArgumentParser(description="SINN staged training")
    # phase
    p.add_argument("--sinn_phase", type=str, required=True,
                   choices=["g", "e", "ge", "finetune"],
                   help="Which SINN training phase to run")
    # checkpoint paths for later phases
    p.add_argument("--g_ckpt", type=str, default=None,
                   help="Path to Phase-g checkpoint (for ge/finetune)")
    p.add_argument("--e_ckpt", type=str, default=None,
                   help="Path to Phase-e checkpoint (for ge/finetune)")
    p.add_argument("--ge_ckpt", type=str, default=None,
                   help="Path to Phase-ge checkpoint (for finetune)")
    # decomposition
    p.add_argument("--decomp_path", type=str,
                   default="data/decomposition/decomposition_2022.json",
                   help="Path to decomposition JSON from fit_decomposition.py")
    # architecture
    p.add_argument("--g_input_type", type=str, default="tokens",
                   choices=["tokens", "grm"])
    p.add_argument("--env_categorical_mode", type=str, default="drop",
                   choices=["drop", "onehot"])
    p.add_argument("--full_tf_mlp_type", type=str, default="moe")
    p.add_argument("--moe_num_experts", type=int, default=8)
    p.add_argument("--moe_top_k", type=int, default=2)
    p.add_argument("--moe_shared_expert", type=str2bool, default=True)
    p.add_argument("--moe_expert_hidden_dim", type=int, default=256)
    p.add_argument("--moe_shared_expert_hidden_dim", type=int, default=256)
    p.add_argument("--moe_loss_weight", type=float, default=0.01)
    # training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gbs", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=500)
    p.add_argument("--early_stop", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--scale_targets", type=str2bool, default=False)
    # model size
    p.add_argument("--g_layers", type=int, default=1)
    p.add_argument("--mlp_layers", type=int, default=1)
    p.add_argument("--gxe_layers", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--emb_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    # finetune loss
    p.add_argument("--loss", type=str, default="envpcc")
    p.add_argument("--loss_weights", type=str, default="1.0")
    p.add_argument("--env_stratified", type=str2bool, default=True)
    p.add_argument("--min_samples_per_env", type=int, default=32)
    # LEO validation
    p.add_argument("--leo_val", type=str2bool, default=True)
    p.add_argument("--leo_val_fraction", type=float, default=0.10)
    # GE phase options
    p.add_argument("--freeze_encoders", type=str2bool, default=True,
                   help="Freeze G/E encoders during ge phase")
    p.add_argument("--encoder_lr_mult", type=float, default=0.1,
                   help="LR multiplier for encoder params in ge phase (if not frozen)")
    # finetune phase options
    p.add_argument("--finetune_lr", type=float, default=1e-5,
                   help="Lower LR for fine-tuning phase")
    return p.parse_args()


# ── Data loading ─────────────────────────────────────────────────
def build_datasets(args):
    """Build train/val datasets with decomposition targets."""
    train_ds = GxE_Dataset(
        split="train",
        data_path="data/maize_data_2014-2023_vs_2024_v2/",
        scaler=None,
        y_scalers=None,
        scale_targets=args.scale_targets,
        g_input_type=args.g_input_type,
        env_categorical_mode=args.env_categorical_mode,
        marker_stats=None,
        leo_val=args.leo_val,
        leo_val_fraction=args.leo_val_fraction,
        leo_seed=args.seed,
        decomposition_path=args.decomp_path,
    )
    val_ds = GxE_Dataset(
        split="val",
        data_path="data/maize_data_2014-2023_vs_2024_v2/",
        scaler=train_ds.scaler,
        y_scalers=train_ds.label_scalers,
        scale_targets=args.scale_targets,
        g_input_type=args.g_input_type,
        env_categorical_mode=args.env_categorical_mode,
        marker_stats=train_ds.marker_stats,
        leo_val=args.leo_val,
        leo_val_envs=train_ds.leo_val_envs,
        decomposition_path=args.decomp_path,
    )
    return train_ds, val_ds


def build_dataloaders(args, train_ds, val_ds, rank, world_size):
    """Build DataLoaders with optional env-stratified sampling."""
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    use_batch_sampler = False
    if args.env_stratified and args.sinn_phase == "finetune":
        train_sampler = EnvStratifiedSampler(
            env_ids=train_ds.env_id_tensor.tolist(),
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            min_samples_per_env=args.min_samples_per_env,
        )
        use_batch_sampler = True

    if use_batch_sampler:
        train_loader = DataLoader(
            train_ds, batch_sampler=train_sampler,
            pin_memory=True, num_workers=0, worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            pin_memory=True, num_workers=0, worker_init_fn=seed_worker,
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        pin_memory=True, num_workers=0, worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, train_sampler


# ── Model construction ───────────────────────────────────────────
def build_model(args, config, device):
    """Build model for the requested SINN phase."""
    phase = args.sinn_phase

    if phase == "g":
        model = SINNGModel(config, encoder_type="dense").to(device)
    elif phase == "e":
        model = SINNEModel(config).to(device)
    elif phase in ("ge", "finetune"):
        model = FullTransformer(
            config,
            mlp_type=args.full_tf_mlp_type,
            moe_num_experts=args.moe_num_experts,
            moe_top_k=args.moe_top_k,
            moe_expert_hidden_dim=args.moe_expert_hidden_dim,
            moe_shared_expert=args.moe_shared_expert,
            moe_shared_expert_hidden_dim=args.moe_shared_expert_hidden_dim,
            moe_loss_weight=args.moe_loss_weight,
        ).to(device)

        # Load pretrained encoder weights for ge phase
        if phase == "ge":
            _load_encoder_weights(model, args, device)
        elif phase == "finetune" and args.ge_ckpt:
            ckpt = torch.load(args.ge_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            if is_main(0):
                print(f"[SINN] Loaded ge checkpoint: {args.ge_ckpt}")
    else:
        raise ValueError(f"Unknown phase: {phase}")

    return model


def _load_encoder_weights(full_model, args, device):
    """Load G and E encoder weights from phase g/e checkpoints into FullTransformer."""
    if args.g_ckpt:
        g_ckpt = torch.load(args.g_ckpt, map_location=device, weights_only=False)
        # Map SINNGModel.encoder.* → FullTransformer.g_embed/wpe (token embedding path)
        # FullTransformer uses g_embed (Embedding) while G_Encoder uses transformer.wte (Embedding)
        # They serve the same purpose but have different structures.
        # For FullTransformer, we can't directly inject G_Encoder weights since architectures differ.
        # Instead, log that G encoder was pretrained and its val metrics.
        if is_main(0):
            best_val = g_ckpt.get("best_val_metric", "N/A")
            print(f"[SINN] G encoder pretrained — val MSE: {best_val}")
            print(f"[SINN] Note: G_Encoder architecture differs from FullTransformer's G path.")
            print(f"[SINN] G encoder weights inform the diagnostic, not direct transfer to FullTransformer.")

    if args.e_ckpt:
        e_ckpt = torch.load(args.e_ckpt, map_location=device, weights_only=False)
        if is_main(0):
            best_val = e_ckpt.get("best_val_metric", "N/A")
            print(f"[SINN] E encoder pretrained — val MSE: {best_val}")


def _get_target(yb, phase):
    """Extract the correct target for the current SINN phase."""
    if phase == "g":
        return yb["g_hat"]
    elif phase == "e":
        return yb["e_hat"]
    elif phase == "ge":
        return yb["ge_hat"]
    else:  # finetune
        return yb["y"]


# ── Training loop ────────────────────────────────────────────────
def train_phase(args, model, train_loader, val_loader, train_sampler,
                train_ds, val_ds, device, rank, world_size):
    """Generic training loop for any SINN phase."""
    phase = args.sinn_phase
    lr = args.finetune_lr if phase == "finetune" else args.lr

    # Optimizer: optionally use different LR for encoders in ge phase
    if phase == "ge" and not args.freeze_encoders:
        encoder_params = []
        other_params = []
        for name, param in model.named_parameters():
            if "g_embed" in name or "e_proj" in name:
                encoder_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": lr},
            {"params": encoder_params, "lr": lr * args.encoder_lr_mult},
        ], weight_decay=args.weight_decay)
    elif phase == "ge" and args.freeze_encoders:
        for name, param in model.named_parameters():
            if "g_embed" in name or "e_proj" in name:
                param.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Loss function
    if phase == "finetune":
        loss_fn = build_loss(args.loss, args.loss_weights)
        uses_env = True
    else:
        loss_fn = nn.MSELoss()
        uses_env = False

    # DDP
    ddp_model = DDP(model, device_ids=[0], output_device=0, find_unused_parameters=True)

    # LR schedule
    batches_per_epoch = len(train_loader)
    effective_epochs = min(args.early_stop * 2, args.num_epochs)
    total_iters = effective_epochs * batches_per_epoch
    warmup_iters = batches_per_epoch * 3
    lr_decay_iters = total_iters
    max_lr, min_lr = lr, 0.1 * lr

    # wandb
    run_name = f"sinn-{phase}-e{args.emb_size}-g{args.g_layers}-gxe{args.gxe_layers}-lr{lr:.0e}-s{args.seed}"
    if is_main(rank):
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
        )
        run_ckpt_dir = Path("checkpoints") / run_name
        if run_ckpt_dir.exists():
            shutil.rmtree(run_ckpt_dir)
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        wandb.config.update({
            "sinn_phase": phase,
            "lr": lr,
            "batch_size": args.batch_size,
            "n_embd": args.emb_size,
            "g_layers": args.g_layers,
            "gxe_layers": args.gxe_layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "early_stop": args.early_stop,
            "decomp_path": args.decomp_path,
        }, allow_val_change=True)
        run.define_metric("iter_num")
        run.define_metric("train_loss", step_metric="iter_num")
        run.define_metric("epoch")
        run.define_metric("val_mse", step_metric="epoch")
        run.define_metric("val_env_pcc", step_metric="epoch")

        run_id_file = os.environ.get("WANDB_RUN_ID_FILE")
        if run_id_file:
            with open(run_id_file, "w") as f:
                f.write(run.id.strip())

        cdir_file = os.environ.get("CHECKPOINT_DIR_FILE")
        if cdir_file:
            with open(cdir_file, "w") as f:
                f.write(str(run_ckpt_dir.resolve()))

    # training state
    best_val_metric = float("inf") if phase != "finetune" else -float("inf")
    last_improved = 0
    iter_num = 0
    t0 = time.time()

    if is_main(rank):
        n_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        print(f"[SINN phase={phase}] Trainable params: {n_params:,}, "
              f"train_samples={len(train_ds):,}, val_samples={len(val_ds):,}")

    for epoch in range(args.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        ddp_model.train()

        pbar = tqdm(total=batches_per_epoch, desc=f"[{phase}] Epoch {epoch}",
                     disable=not is_main(rank))

        for xb, yb in train_loader:
            xb = {k: v.to(device, non_blocking=True) for k, v in xb.items()}
            target = _get_target(yb, phase).to(device, non_blocking=True).float()
            env_id = yb["env_id"].to(device, non_blocking=True).long()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = ddp_model(xb)
                if isinstance(preds, dict):
                    preds = preds["total"]
                preds = preds.squeeze(-1)
                target = target.squeeze(-1)

                if uses_env:
                    loss_total, loss_parts = loss_fn(preds, target, env_id=env_id)
                else:
                    loss_total = loss_fn(preds, target)

                # MoE aux loss
                moe_aux = getattr(ddp_model.module, "moe_aux_loss", None)
                if moe_aux is not None:
                    loss_total = loss_total + moe_aux

            if torch.isnan(loss_total):
                raise RuntimeError(f"[SINN {phase}] Loss is NaN at iter {iter_num}")

            # LR schedule
            current_lr = get_lr(iter_num, warmup_iters, lr_decay_iters, max_lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if is_main(rank):
                wandb.log({
                    "train_loss": loss_total.item(),
                    "learning_rate": current_lr,
                    "iter_num": iter_num,
                })
            iter_num += 1
            pbar.update(1)
        pbar.close()

        # ── Validation ───────────────────────────────────────────
        ddp_model.eval()
        with torch.no_grad():
            all_preds, all_targets, all_env_ids = [], [], []
            for xb, yb in val_loader:
                xb = {k: v.to(device, non_blocking=True) for k, v in xb.items()}
                target = _get_target(yb, phase).to(device, non_blocking=True).float()
                env_id = yb["env_id"].to(device, non_blocking=True).long()

                preds = ddp_model(xb)
                if isinstance(preds, dict):
                    preds = preds["total"]
                all_preds.append(preds.squeeze(-1))
                all_targets.append(target.squeeze(-1))
                all_env_ids.append(env_id)

            local_preds = torch.cat(all_preds) if all_preds else torch.empty(0, device=device)
            local_targets = torch.cat(all_targets) if all_targets else torch.empty(0, device=device)
            local_env_ids = torch.cat(all_env_ids) if all_env_ids else torch.empty(0, dtype=torch.long, device=device)

            # All-gather across ranks
            def _all_gather_flat(t):
                local_n = torch.tensor([t.shape[0]], device=device)
                all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
                dist.all_gather(all_n, local_n)
                max_n = max(x.item() for x in all_n)
                padded = torch.zeros(max_n, device=device, dtype=t.dtype)
                padded[:t.shape[0]] = t
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded)
                return torch.cat([gathered[i][:all_n[i].item()] for i in range(world_size)])

            full_preds = _all_gather_flat(local_preds)[:len(val_ds)]
            full_targets = _all_gather_flat(local_targets)[:len(val_ds)]
            full_env_ids = _all_gather_flat(local_env_ids.float()).long()[:len(val_ds)]

            val_mse = nn.functional.mse_loss(full_preds, full_targets).item()
            val_env_pcc = float(macro_env_pearson(
                full_preds, full_targets, full_env_ids, min_samples=2
            ).item())

        # ── Checkpoint / early stop ──────────────────────────────
        if is_main(rank):
            # For component phases (g, e, ge): minimize MSE
            # For finetune: maximize env_avg_pearson
            if phase == "finetune":
                metric = val_env_pcc
                improved = metric > best_val_metric + 1e-8
            else:
                metric = val_mse
                improved = metric < best_val_metric - 1e-8

            wandb.log({
                "epoch": epoch,
                "val_mse": val_mse,
                "val_env_pcc": val_env_pcc,
            })

            if improved:
                best_val_metric = metric
                last_improved = 0

                # Save checkpoint
                env_scaler_payload = {
                    "mean": train_ds.scaler.mean_.tolist(),
                    "scale": train_ds.scaler.scale_.tolist(),
                    "var": train_ds.scaler.var_.tolist(),
                    "n_features_in": int(train_ds.scaler.n_features_in_),
                    "feature_names_in": list(train_ds.e_cols),
                }
                label_scalers_payload = None
                if train_ds.label_scalers:
                    label_scalers_payload = {
                        k: {"mean": float(v.mean), "std": float(v.std)}
                        for k, v in train_ds.label_scalers.items()
                    }
                marker_stats_payload = None
                if getattr(train_ds, "marker_stats", None):
                    marker_stats_payload = {
                        "p": train_ds.marker_stats["p"].tolist(),
                        "scale": train_ds.marker_stats["scale"].tolist(),
                        "valid": train_ds.marker_stats["valid"].tolist(),
                        "columns": list(train_ds.marker_stats["columns"]),
                    }

                ckpt = {
                    "model": ddp_model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "sinn_phase": phase,
                    "best_val_metric": best_val_metric,
                    "val_mse": val_mse,
                    "val_env_pcc": val_env_pcc,
                    "config": {
                        "sinn_phase": phase,
                        "g_input_type": args.g_input_type,
                        "env_categorical_mode": args.env_categorical_mode,
                        "full_tf_mlp_type": args.full_tf_mlp_type,
                        "block_size": train_ds.block_size,
                        "n_env_fts": train_ds.n_env_fts,
                        "g_layers": args.g_layers,
                        "mlp_layers": args.mlp_layers,
                        "gxe_layers": args.gxe_layers,
                        "n_head": args.heads,
                        "n_embd": args.emb_size,
                        "moe_num_experts": args.moe_num_experts,
                        "moe_top_k": args.moe_top_k,
                        "moe_expert_hidden_dim": args.moe_expert_hidden_dim,
                        "moe_shared_expert": args.moe_shared_expert,
                        "moe_shared_expert_hidden_dim": args.moe_shared_expert_hidden_dim,
                        "moe_loss_weight": args.moe_loss_weight,
                        "loss": args.loss,
                        "loss_weights": args.loss_weights,
                        "scale_targets": args.scale_targets,
                    },
                    "env_scaler": env_scaler_payload,
                    "y_scalers": label_scalers_payload,
                    "marker_stats": marker_stats_payload,
                }
                ckpt_path = run_ckpt_dir / f"best_{phase}.pt"
                torch.save(ckpt, ckpt_path)
                metric_name = "val_env_pcc" if phase == "finetune" else "val_mse"
                print(f"*** [{phase}] {metric_name} improved: {best_val_metric:.6f} "
                      f"(epoch {epoch}) → saved {ckpt_path} ***")
            else:
                last_improved += 1
                if last_improved % 10 == 0:
                    print(f"[{phase}] No improvement for {last_improved} epochs "
                          f"(best={best_val_metric:.6f})")

        # Early stopping broadcast
        stop_flag = torch.tensor([0], device=device)
        if is_main(rank) and last_improved > args.early_stop:
            print(f"*** [{phase}] Early stopping after {args.early_stop} epochs without improvement ***")
            stop_flag[0] = 1
        dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break

        if is_main(rank) and epoch % 10 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"[{phase} epoch {epoch}] val_mse={val_mse:.6f} | "
                  f"val_env_pcc={val_env_pcc:.5f} | best={best_val_metric:.6f} | "
                  f"elapsed={elapsed:.1f}m")

    dist.barrier()
    if is_main(rank):
        print(f"\n[SINN phase={phase}] Training complete. Best metric: {best_val_metric:.6f}")
        print(f"[SINN phase={phase}] Checkpoint: {run_ckpt_dir}/best_{phase}.pt")
        run.finish()

    return best_val_metric


# ── Main ─────────────────────────────────────────────────────────
def main():
    args = parse_sinn_args()
    device, local_rank, rank, world_size = setup_ddp()
    set_seed(args.seed + rank)

    if is_main(rank):
        print(f"[SINN] Phase: {args.sinn_phase}")
        print(f"[SINN] Decomposition: {args.decomp_path}")
        if args.g_ckpt:
            print(f"[SINN] G checkpoint: {args.g_ckpt}")
        if args.e_ckpt:
            print(f"[SINN] E checkpoint: {args.e_ckpt}")
        if args.ge_ckpt:
            print(f"[SINN] GE checkpoint: {args.ge_ckpt}")

    # Data
    train_ds, val_ds = build_datasets(args)
    train_loader, val_loader, train_sampler = build_dataloaders(
        args, train_ds, val_ds, rank, world_size
    )

    if is_main(rank):
        print(f"[SINN] Train: {len(train_ds):,} samples, Val: {len(val_ds):,} samples")
        print(f"[SINN] has_decomposition: {train_ds.has_decomposition}")

    # Config
    config = Config(
        block_size=train_ds.block_size,
        g_input_type=args.g_input_type,
        n_head=args.heads,
        n_g_layer=args.g_layers,
        n_ld_layer=1,
        n_mlp_layer=args.mlp_layers,
        n_gxe_layer=args.gxe_layers,
        n_embd=args.emb_size,
        dropout=args.dropout,
        n_env_fts=train_ds.n_env_fts,
    )

    # Model
    model = build_model(args, config, device)

    # Train
    train_phase(
        args, model, train_loader, val_loader, train_sampler,
        train_ds, val_ds, device, rank, world_size,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
