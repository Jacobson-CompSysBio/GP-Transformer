"""
SINN-style staged training for GP-Transformer.

Phases:
  g        — Train G encoder on G_hat targets (genotype main effect)
  e        — Train E encoder on E_hat targets (environment main effect)
  ge       — Two-prong model: pretrained G_Encoder + E_Encoder fused via
             self-attention, trained on GE_hat targets (interaction residual)
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
from models.model import FullTransformer  # kept for eval compatibility
from models.transformer import G_Encoder, TransformerBlock
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
        e_layers = getattr(config, "e_mlp_layers", config.n_mlp_layer)
        self.encoder = E_Encoder(
            input_dim=config.n_env_fts,
            output_dim=config.n_embd,
            hidden_dim=config.n_embd,
            n_hidden=e_layers,
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


class SINNGxEModel(nn.Module):
    """Two-prong GxE: pretrained G_Encoder + E_Encoder → self-attention fusion → prediction.

    Includes auxiliary G/E heads that can be used as regularizers during finetune
    to keep encoder representations faithful to the decomposition.
    """

    def __init__(self, config, n_gxe_layers=1, **encoder_kwargs):
        super().__init__()
        self.g_encoder = G_Encoder(config, **encoder_kwargs)
        e_layers = getattr(config, "e_mlp_layers", config.n_mlp_layer)
        self.e_encoder = E_Encoder(
            input_dim=config.n_env_fts,
            output_dim=config.n_embd,
            hidden_dim=config.n_embd,
            n_hidden=e_layers,
            dropout=config.dropout,
        )
        # GxE interaction: self-attention over concatenated G + E tokens
        self.gxe_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(n_gxe_layers)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

        # Auxiliary heads: predict G_hat / E_hat from encoder representations.
        # These don't participate in the main prediction path — they act as
        # regularizers during finetune to anchor encoder outputs to decomposition.
        self.g_aux_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1),
        )
        self.e_aux_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 4),
            nn.GELU(),
            nn.Linear(config.n_embd // 4, 1),
        )
        nn.init.normal_(self.g_aux_head[-1].weight, std=0.01)
        nn.init.zeros_(self.g_aux_head[-1].bias)
        nn.init.normal_(self.e_aux_head[-1].weight, std=0.01)
        nn.init.zeros_(self.e_aux_head[-1].bias)

    def forward(self, x, return_aux=False):
        g_enc = self.g_encoder(x["g_data"])   # (B, T+1, C) with CLS at [0]
        e_enc = self.e_encoder(x["e_data"])   # (B, C)
        e_tok = e_enc.unsqueeze(1)             # (B, 1, C)
        tokens = torch.cat([g_enc, e_tok], dim=1)  # (B, T+2, C)
        for block in self.gxe_blocks:
            tokens = block(tokens)
        tokens = self.ln_f(tokens)
        cls_repr = tokens[:, 0]               # CLS token → (B, C)
        pred = self.head(cls_repr)             # (B, 1)

        if not return_aux:
            return pred

        # Auxiliary predictions from pre-attention encoder representations
        g_cls = g_enc[:, 0]                    # (B, C) — G CLS before GxE attention
        g_aux = self.g_aux_head(g_cls)         # (B, 1)
        e_aux = self.e_aux_head(e_enc)         # (B, 1)
        return pred, g_aux, e_aux


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
    p.add_argument("--e_mlp_layers", type=int, default=None,
                   help="E encoder MLP depth (defaults to --mlp_layers if not set)")
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
    p.add_argument("--freeze_encoders", type=str2bool, default=False,
                   help="Freeze G/E encoders during ge phase")
    p.add_argument("--ge_encoder_lr", type=float, default=None,
                   help="Explicit LR for encoder params in ge phase (if not frozen). "
                        "Falls back to lr * encoder_lr_mult if not set.")
    p.add_argument("--encoder_lr_mult", type=float, default=0.1,
                   help="LR multiplier for encoder params in ge phase (if not frozen)")
    # finetune phase options
    p.add_argument("--finetune_lr", type=float, default=1e-5,
                   help="Lower LR for fine-tuning phase")
    # auxiliary decomposition loss weights (finetune regularization)
    p.add_argument("--aux_g_weight", type=float, default=0.0,
                   help="Weight for G_hat auxiliary MSE loss during finetune")
    p.add_argument("--aux_e_weight", type=float, default=0.0,
                   help="Weight for E_hat auxiliary MSE loss during finetune")
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
        model = SINNGxEModel(
            config,
            n_gxe_layers=args.gxe_layers,
            encoder_type="dense",
        ).to(device)

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


def _load_encoder_weights(model, args, device):
    """Load G and E encoder weights from phase g/e checkpoints into SINNGxEModel."""
    if args.g_ckpt:
        g_ckpt = torch.load(args.g_ckpt, map_location=device, weights_only=False)
        g_state = g_ckpt["model"]
        # SINNGModel saves as encoder.* → load into SINNGxEModel.g_encoder.*
        g_encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in g_state.items() if k.startswith("encoder.")
        }
        missing, unexpected = model.g_encoder.load_state_dict(g_encoder_state, strict=False)
        if is_main(0):
            best_val = g_ckpt.get("best_val_metric", "N/A")
            print(f"[SINN] Loaded G encoder weights (val MSE: {best_val})")
            if missing:
                print(f"[SINN]   G missing keys: {missing}")
            if unexpected:
                print(f"[SINN]   G unexpected keys: {unexpected}")

    if args.e_ckpt:
        e_ckpt = torch.load(args.e_ckpt, map_location=device, weights_only=False)
        e_state = e_ckpt["model"]
        # SINNEModel saves as encoder.* → load into SINNGxEModel.e_encoder.*
        e_encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in e_state.items() if k.startswith("encoder.")
        }
        missing, unexpected = model.e_encoder.load_state_dict(e_encoder_state, strict=False)
        if is_main(0):
            best_val = e_ckpt.get("best_val_metric", "N/A")
            print(f"[SINN] Loaded E encoder weights (val MSE: {best_val})")
            if missing:
                print(f"[SINN]   E missing keys: {missing}")
            if unexpected:
                print(f"[SINN]   E unexpected keys: {unexpected}")


def _env_level_pearson(preds, targets, env_ids, eps=1e-8):
    """Pearson correlation across environments (one mean-pred per env vs target).

    For environment-level targets like E_hat (constant within env),
    macro_env_pearson returns NaN because within-env variance is zero.
    This function instead aggregates to one value per env and correlates across envs.
    """
    env_preds, env_targets = [], []
    for eid in torch.unique(env_ids):
        mask = env_ids == eid
        env_preds.append(preds[mask].mean())
        env_targets.append(targets[mask][0])  # constant within env
    if len(env_preds) < 3:
        return float("nan")
    p = torch.stack(env_preds).float()
    t = torch.stack(env_targets).float()
    p_c = p - p.mean()
    t_c = t - t.mean()
    num = (p_c * t_c).sum()
    den = (p_c.norm() * t_c.norm())
    if den < eps:
        return float("nan")
    return float((num / den).clamp(-1, 1).item())


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
            if name.startswith("g_encoder.") or name.startswith("e_encoder."):
                encoder_params.append(param)
            else:
                other_params.append(param)
        enc_lr = args.ge_encoder_lr if args.ge_encoder_lr is not None else lr * args.encoder_lr_mult
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": lr, "initial_lr": lr},
            {"params": encoder_params, "lr": enc_lr, "initial_lr": enc_lr},
        ], weight_decay=args.weight_decay)
    elif phase == "ge" and args.freeze_encoders:
        for name, param in model.named_parameters():
            if name.startswith("g_encoder.") or name.startswith("e_encoder."):
                param.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)

    # Store initial_lr for all param groups (used by cosine schedule)
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    # Loss function
    if phase == "finetune":
        loss_fn = build_loss(args.loss, args.loss_weights)
        uses_env = True
    else:
        loss_fn = nn.MSELoss()
        uses_env = False

    # Auxiliary decomposition losses for finetune regularization
    use_aux = phase == "finetune" and (args.aux_g_weight > 0 or args.aux_e_weight > 0)
    aux_mse_fn = nn.MSELoss() if use_aux else None

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
            project="gxe-transformer",
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
            "aux_g_weight": args.aux_g_weight,
            "aux_e_weight": args.aux_e_weight,
        }, allow_val_change=True)
        run.define_metric("iter_num")
        run.define_metric("train_loss", step_metric="iter_num")
        run.define_metric("epoch")
        run.define_metric("val/mse", step_metric="epoch")
        run.define_metric("val/env_pcc", step_metric="epoch")
        run.define_metric("val/G_enc_loss", step_metric="epoch")
        run.define_metric("val/E_enc_loss", step_metric="epoch")
        run.define_metric("val/GxE_loss", step_metric="epoch")

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
                if use_aux:
                    preds, g_aux_pred, e_aux_pred = ddp_model(xb, return_aux=True)
                else:
                    preds = ddp_model(xb)
                if isinstance(preds, dict):
                    preds = preds["total"]
                preds = preds.squeeze(-1)
                target = target.squeeze(-1)

                # Mask novel-env rows for E-phase (E_hat=0 is meaningless)
                if phase == "e" and "has_e_hat" in yb:
                    mask = yb["has_e_hat"].to(device, non_blocking=True).squeeze(-1)
                    if mask.any():
                        preds = preds[mask]
                        target = target[mask]
                    else:
                        # Skip batch entirely if no valid targets
                        pbar.update(1)
                        iter_num += 1
                        continue

                if uses_env:
                    loss_total, loss_parts = loss_fn(preds, target, env_id=env_id)
                else:
                    loss_total = loss_fn(preds, target)

                # Auxiliary decomposition regularization
                if use_aux:
                    if args.aux_g_weight > 0:
                        g_hat_target = yb["g_hat"].to(device, non_blocking=True).float().squeeze(-1)
                        loss_g_aux = aux_mse_fn(g_aux_pred.squeeze(-1), g_hat_target)
                        loss_total = loss_total + args.aux_g_weight * loss_g_aux
                    if args.aux_e_weight > 0:
                        e_hat_target = yb["e_hat"].to(device, non_blocking=True).float().squeeze(-1)
                        loss_e_aux = aux_mse_fn(e_aux_pred.squeeze(-1), e_hat_target)
                        loss_total = loss_total + args.aux_e_weight * loss_e_aux

                # MoE aux loss
                moe_aux = getattr(ddp_model.module, "moe_aux_loss", None)
                if moe_aux is not None:
                    loss_total = loss_total + moe_aux

            if torch.isnan(loss_total):
                raise RuntimeError(f"[SINN {phase}] Loss is NaN at iter {iter_num}")

            # LR schedule — preserve ratio between param groups for differential LR
            current_lr = get_lr(iter_num, warmup_iters, lr_decay_iters, max_lr, min_lr)
            lr_scale = current_lr / max_lr if max_lr > 0 else 1.0
            for pg in optimizer.param_groups:
                pg["lr"] = pg.get("initial_lr", max_lr) * lr_scale

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if is_main(rank):
                log_dict = {
                    "train_loss": loss_total.item(),
                    "learning_rate": current_lr,
                    "iter_num": iter_num,
                }
                if use_aux and args.aux_g_weight > 0:
                    log_dict["aux_g_mse"] = loss_g_aux.item()
                if use_aux and args.aux_e_weight > 0:
                    log_dict["aux_e_mse"] = loss_e_aux.item()
                wandb.log(log_dict)
            iter_num += 1
            pbar.update(1)
        pbar.close()

        # ── Validation ───────────────────────────────────────────
        ddp_model.eval()
        # Only use aux heads during finetune (they're untrained during GE)
        has_aux_model = phase == "finetune" and hasattr(ddp_model.module, "g_aux_head")
        with torch.no_grad():
            all_preds, all_targets, all_env_ids = [], [], []
            all_g_hat, all_e_hat, all_g_aux, all_e_aux = [], [], [], []
            all_has_e_hat = []
            for xb, yb in val_loader:
                xb = {k: v.to(device, non_blocking=True) for k, v in xb.items()}
                target = _get_target(yb, phase).to(device, non_blocking=True).float()
                env_id = yb["env_id"].to(device, non_blocking=True).long()

                if has_aux_model:
                    preds, g_aux, e_aux = ddp_model(xb, return_aux=True)
                    all_g_aux.append(g_aux.squeeze(-1))
                    all_e_aux.append(e_aux.squeeze(-1))
                    all_g_hat.append(yb["g_hat"].to(device).float().squeeze(-1))
                    all_e_hat.append(yb["e_hat"].to(device).float().squeeze(-1))
                else:
                    preds = ddp_model(xb)
                if isinstance(preds, dict):
                    preds = preds["total"]
                all_preds.append(preds.squeeze(-1))
                all_targets.append(target.squeeze(-1))
                all_env_ids.append(env_id)
                if "has_e_hat" in yb:
                    all_has_e_hat.append(yb["has_e_hat"].to(device).squeeze(-1))

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
            # For phases with env-level targets (E_hat is constant per env),
            # macro_env_pearson is undefined (zero within-env variance).
            # Compute env-level PCC instead: mean pred per env vs E_hat.
            if phase == "e":
                val_env_pcc = _env_level_pearson(full_preds, full_targets, full_env_ids)
            else:
                val_env_pcc = float(macro_env_pearson(
                    full_preds, full_targets, full_env_ids, min_samples=2
                ).item())

            # Per-encoder validation metrics (GE / finetune phases)
            val_g_enc_mse = float("nan")
            val_e_enc_mse = float("nan")
            if has_aux_model and all_g_aux:
                full_g_aux = _all_gather_flat(torch.cat(all_g_aux))[:len(val_ds)]
                full_e_aux = _all_gather_flat(torch.cat(all_e_aux))[:len(val_ds)]
                full_g_hat = _all_gather_flat(torch.cat(all_g_hat))[:len(val_ds)]
                full_e_hat = _all_gather_flat(torch.cat(all_e_hat))[:len(val_ds)]
                val_g_enc_mse = nn.functional.mse_loss(full_g_aux, full_g_hat).item()
                # E encoder loss only on rows with valid E_hat
                if all_has_e_hat:
                    full_has_e = _all_gather_flat(torch.cat(all_has_e_hat).float())[:len(val_ds)].bool()
                    if full_has_e.any():
                        val_e_enc_mse = nn.functional.mse_loss(
                            full_e_aux[full_has_e], full_e_hat[full_has_e]
                        ).item()
                else:
                    val_e_enc_mse = nn.functional.mse_loss(full_e_aux, full_e_hat).item()

            # For phase E: compute val_mse only on valid (non-novel) rows
            if phase == "e" and all_has_e_hat:
                full_has_e = _all_gather_flat(torch.cat(all_has_e_hat).float())[:len(val_ds)].bool()
                if full_has_e.any():
                    val_mse = nn.functional.mse_loss(
                        full_preds[full_has_e], full_targets[full_has_e]
                    ).item()

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

            val_log = {
                "epoch": epoch,
                "val/mse": val_mse,
                "val/env_pcc": val_env_pcc,
            }
            # Per-encoder tracking
            if phase == "g":
                val_log["val/G_enc_loss"] = val_mse
            elif phase == "e":
                val_log["val/E_enc_loss"] = val_mse
            elif phase in ("ge", "finetune"):
                if not math.isnan(val_g_enc_mse):
                    val_log["val/G_enc_loss"] = val_g_enc_mse
                if not math.isnan(val_e_enc_mse):
                    val_log["val/E_enc_loss"] = val_e_enc_mse
                val_log["val/GxE_loss"] = val_mse
            wandb.log(val_log)

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
                        "e_mlp_layers": args.e_mlp_layers if args.e_mlp_layers is not None else args.mlp_layers,
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
            extra = ""
            if has_aux_model and not math.isnan(val_g_enc_mse):
                extra += f" | G_enc={val_g_enc_mse:.4f}"
            if has_aux_model and not math.isnan(val_e_enc_mse):
                extra += f" | E_enc={val_e_enc_mse:.4f}"
            print(f"[{phase} epoch {epoch}] val_mse={val_mse:.6f} | "
                  f"val_env_pcc={val_env_pcc:.5f} | best={best_val_metric:.6f}{extra} | "
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
    e_mlp_layers = args.e_mlp_layers if args.e_mlp_layers is not None else args.mlp_layers
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
    config.e_mlp_layers = e_mlp_layers

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
