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
from utils.loss import (
    build_loss,
    GlobalPearsonCorrLoss,
    GenomicContrastiveLoss,
    EnvironmentContrastiveLoss,
    compute_ibs_similarity,
    compute_grm_similarity,
    macro_env_pearson,
    macro_env_ccc,
)
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

    # Ensure MASTER_ADDR is set and routable (hostname -i can return IPv6 on Frontier)
    master = os.environ.get("MASTER_ADDR", "")
    if not master or ":" in master:  # missing or IPv6
        master = extract_master_addr()
        os.environ["MASTER_ADDR"] = master
    if rank == 0:
        print(f"[DDP] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}, world_size={world_size}")

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

def _normalize_choice(name: str, value, allowed: set[str], default: str) -> str:
    """
    Backward-compatible normalizer for legacy boolean-like CLI values.
    """
    v = str(value).strip().lower()
    if v in {"", "false", "0", "off", "none", "no"}:
        return default
    if v not in allowed:
        raise ValueError(f"Unsupported {name}='{value}'. Allowed: {sorted(allowed)}")
    return v


def _ramp_factor(epoch_num: int, start_epoch: int, ramp_epochs: int) -> float:
    if epoch_num < int(start_epoch):
        return 0.0
    if int(ramp_epochs) <= 0:
        return 1.0
    return min(1.0, float(epoch_num - int(start_epoch)) / float(ramp_epochs))


def _extract_prediction_tensors(preds):
    if isinstance(preds, dict):
        total_pred = preds.get("total")
        rank_pred = preds.get("rank", total_pred)
        if total_pred is None:
            total_pred = rank_pred
        return total_pred, rank_pred
    return preds, preds


def _proxy_support_fraction(train_ds, proxy_ds, proxy_tester: str) -> float:
    if proxy_ds is None or len(proxy_ds) == 0:
        return float("nan")
    if not hasattr(train_ds, "meta") or not hasattr(proxy_ds, "meta"):
        return float("nan")
    if "parent1" not in train_ds.meta.columns or "parent1" not in proxy_ds.meta.columns:
        return float("nan")

    heldout_parent1 = set(proxy_ds.meta['parent1'].astype(str).unique().tolist())
    if not heldout_parent1:
        return float("nan")
    support_parent1 = set(
        train_ds.meta.loc[train_ds.meta['parent2'].astype(str) != str(proxy_tester), 'parent1']
        .astype(str)
        .unique()
        .tolist()
    )
    return len(heldout_parent1 & support_parent1) / max(1, len(heldout_parent1))


def _debug_sync_probe(enabled: bool, rank: int, label: str, tensor: torch.Tensor | None = None):
    if not enabled or not is_main(rank):
        return
    torch.cuda.synchronize()
    msg = f"[DEBUG_PROBE] {label}"
    if tensor is not None and isinstance(tensor, torch.Tensor):
        with torch.no_grad():
            msg += (
                f" shape={tuple(tensor.shape)}"
                f" dtype={tensor.dtype}"
                f" finite={bool(torch.isfinite(tensor).all().item())}"
            )
    print(msg, flush=True)


def _metric_improved(current_score: float,
                     best_score: float,
                     tie_breakers: list[tuple[float, float]] | None = None,
                     eps: float = 1e-8) -> bool:
    if not math.isfinite(current_score):
        return False
    if not math.isfinite(best_score):
        return True
    if current_score > best_score + eps:
        return True
    if abs(current_score - best_score) > eps:
        return False
    for current_tie, best_tie in tie_breakers or []:
        if current_tie < best_tie - eps:
            return True
        if abs(current_tie - best_tie) > eps:
            return False
    return False


def _safe_metric_snapshot(**metrics):
    snapshot = {}
    for key, value in metrics.items():
        if value is None:
            snapshot[key] = None
        elif isinstance(value, (int, float)):
            snapshot[key] = float(value) if math.isfinite(float(value)) else None
        else:
            snapshot[key] = value
    return snapshot


def _update_checkpoint_manifest(manifest_path: Path,
                                alias_state: dict,
                                alias: str,
                                ckpt_path: Path,
                                epoch: int,
                                metrics: dict):
    alias_state[alias] = {
        "path": ckpt_path.name,
        "epoch": int(epoch),
        "metrics": metrics,
    }
    payload = {"aliases": alias_state}
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

### main ###
def main():
    # setup
    args = parse_args()
    base_run_name = make_run_name(args)
    wandb_run_name = add_runtime_suffix(base_run_name, getattr(args, "seed", None))

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
    set_seed(args.seed + rank)  # different seed for each rank
    g = torch.Generator()
    g.manual_seed(args.seed + rank)

    # Check if using LEO (Leave-Environment-Out) validation
    leo_val = _get_arg_or_env("leo_val", "LEO_VAL", False, str2bool)
    leo_val_fraction = _get_arg_or_env("leo_val_fraction", "LEO_VAL_FRACTION", 0.15, float)
    val_scheme = normalize_val_scheme(
        _get_arg_or_env("val_scheme", "VAL_SCHEME", getattr(args, "val_scheme", None), str),
        leo_val=leo_val,
    )
    g_input_type = str(_get_arg_or_env("g_input_type", "G_INPUT_TYPE", "tokens", str)).lower()
    env_categorical_mode = normalize_env_categorical_mode(
        _get_arg_or_env("env_categorical_mode", "ENV_CATEGORICAL_MODE", "drop", str)
    )
    proxy_tester = str(_get_arg_or_env("proxy_tester", "PROXY_TESTER", args.proxy_tester, str))
    proxy_holdout_frac = float(_get_arg_or_env("proxy_holdout_frac", "PROXY_HOLDOUT_FRAC", args.proxy_holdout_frac, float))
    proxy_seed = int(_get_arg_or_env("proxy_seed", "PROXY_SEED", args.proxy_seed, int))
    proxy_disjoint_from_leo = _get_arg_or_env(
        "proxy_disjoint_from_leo",
        "PROXY_DISJOINT_FROM_LEO",
        True,
        str2bool,
    )
    checkpoint_metric = _normalize_choice(
        "checkpoint_metric",
        _get_arg_or_env("checkpoint_metric", "CHECKPOINT_METRIC", "leo", str),
        {"leo", "select", "proxy_scale"},
        "leo",
    )
    use_proxy_split = val_scheme in {"proxy_same_tester", "hybrid_combo"}
    if val_scheme != "hybrid_combo" and checkpoint_metric != "leo":
        if is_main(rank):
            print(f"[WARN] checkpoint_metric={checkpoint_metric} only applies to hybrid_combo; falling back to leo.")
        checkpoint_metric = "leo"

    if is_main(rank):
        print(f"[INFO] Validation scheme: {val_scheme}")
        if val_scheme in {"leo", "hybrid_combo"}:
            print(f"[INFO] LEO fraction: {leo_val_fraction}")
        if use_proxy_split:
            print(
                f"[INFO] Proxy split: tester={proxy_tester}, "
                f"holdout_frac={proxy_holdout_frac:.2f}, seed={proxy_seed}, "
                f"disjoint_from_leo={bool(proxy_disjoint_from_leo)}"
            )
        print(f"[INFO] Canonical checkpoint metric: {checkpoint_metric}")
        print(f"[INFO] Env categorical mode: {env_categorical_mode}")

    # data (samplers are needed for DDP)
    train_ds = GxE_Dataset(
        split="train",
        data_path='data/maize_data_2014-2023_vs_2024_v2/',
        scaler=None,
        y_scalers=None, # train will fit the scalers
        scale_targets=args.scale_targets,
        g_input_type=g_input_type,
        env_categorical_mode=env_categorical_mode,
        marker_stats=None,
        val_scheme=val_scheme,
        leo_val=leo_val,
        leo_val_envs=None,
        leo_val_fraction=leo_val_fraction,
        leo_seed=args.seed,
        proxy_tester=proxy_tester,
        proxy_holdout_frac=proxy_holdout_frac,
        proxy_seed=proxy_seed,
        proxy_split_info=None,
        proxy_disjoint_from_leo=proxy_disjoint_from_leo,
    )
    env_scaler = train_ds.scaler
    y_scalers = train_ds.label_scalers
    marker_stats = train_ds.marker_stats
    leo_val_envs = train_ds.leo_val_envs  # Pass to val_ds for consistency
    proxy_split_info = train_ds.proxy_split_info

    if is_main(rank):
        print(f"[INFO] Train samples: {len(train_ds):,}, Train envs: {train_ds.env_id_tensor.unique().numel()}")
        if leo_val_envs:
            print(f"[INFO] LEO val env count: {len(leo_val_envs)}")

    val_ds = GxE_Dataset(
        split="val",
        data_path="data/maize_data_2014-2023_vs_2024_v2/",
        scaler=env_scaler,
        y_scalers=y_scalers,
        scale_targets=args.scale_targets,
        g_input_type=g_input_type,
        env_categorical_mode=env_categorical_mode,
        marker_stats=marker_stats,
        val_scheme=val_scheme,
        leo_val=leo_val,
        leo_val_envs=leo_val_envs,  # Use same held-out envs computed by train
        leo_val_fraction=leo_val_fraction,
        leo_seed=args.seed,
        proxy_tester=proxy_tester,
        proxy_holdout_frac=proxy_holdout_frac,
        proxy_seed=proxy_seed,
        proxy_split_info=proxy_split_info,
        proxy_disjoint_from_leo=proxy_disjoint_from_leo,
    )

    if is_main(rank):
        print(f"[INFO] Val samples: {len(val_ds):,}, Val envs: {val_ds.env_id_tensor.unique().numel()}")

    proxy_ds = None
    if val_scheme == "hybrid_combo":
        proxy_ds = GxE_Dataset(
            split="proxy_val",
            data_path="data/maize_data_2014-2023_vs_2024_v2/",
            scaler=env_scaler,
            y_scalers=y_scalers,
            scale_targets=args.scale_targets,
            g_input_type=g_input_type,
            env_categorical_mode=env_categorical_mode,
            marker_stats=marker_stats,
            val_scheme=val_scheme,
            leo_val=leo_val,
            leo_val_envs=leo_val_envs,
            leo_val_fraction=leo_val_fraction,
            leo_seed=args.seed,
            proxy_tester=proxy_tester,
            proxy_holdout_frac=proxy_holdout_frac,
            proxy_seed=proxy_seed,
            proxy_split_info=proxy_split_info,
            proxy_disjoint_from_leo=proxy_disjoint_from_leo,
        )
        if is_main(rank):
            proxy_support_fraction = _proxy_support_fraction(train_ds, proxy_ds, proxy_tester)
            overlap_rows = 0
            if "id" in val_ds.meta.columns and "id" in proxy_ds.meta.columns:
                overlap_rows = len(set(val_ds.meta["id"].astype(str)) & set(proxy_ds.meta["id"].astype(str)))
            if proxy_disjoint_from_leo and overlap_rows != 0:
                raise RuntimeError(
                    f"proxy_disjoint_from_leo=True but found {overlap_rows} overlapping val/proxy rows."
                )
            print(
                f"[INFO] Proxy diagnostics: rows={len(proxy_ds):,}, "
                f"hybrids={proxy_ds.meta['Hybrid'].astype(str).nunique()}, "
                f"envs={proxy_ds.meta['Env'].astype(str).nunique()}, "
                f"years={proxy_ds.meta['Year'].value_counts().sort_index().to_dict()}, "
                f"support_fraction={proxy_support_fraction:.3f}, "
                f"leo_overlap_rows={overlap_rows}"
            )

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    proxy_sampler = DistributedSampler(proxy_ds, shuffle=False) if proxy_ds is not None else None

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
            worker_init_fn=seed_worker,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=train_sampler,
            pin_memory=True,
            num_workers=0,
            worker_init_fn=seed_worker,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    proxy_loader = None
    if proxy_ds is not None:
        proxy_loader = DataLoader(
            proxy_ds,
            batch_size=args.batch_size,
            sampler=proxy_sampler,
            pin_memory=True,
            num_workers=0,
            worker_init_fn=seed_worker,
        )

    # set up config
    config = Config(block_size=train_ds.block_size,
                    g_input_type=g_input_type,
                    n_head=args.heads,
                    n_g_layer=args.g_layers,
                    n_ld_layer=args.ld_layers,
                    n_mlp_layer=args.mlp_layers,
                    n_gxe_layer=args.gxe_layers,
                    n_embd=args.emb_size,
                    dropout=args.dropout,
                    n_env_fts=train_ds.n_env_fts)
    config.calibration_mode = getattr(args, "calibration_mode", "none")
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
    calibration_enabled = getattr(args, "calibration_mode", "none") == "env_affine"
    debug_probe = bool(getattr(args, "debug_probe", False))
    debug_max_steps = int(getattr(args, "debug_max_steps", 0) or 0)
    debug_skip_backward = bool(getattr(args, "debug_skip_backward", False))
    debug_skip_optimizer = bool(getattr(args, "debug_skip_optimizer", False))
    debug_no_autocast = bool(getattr(args, "debug_no_autocast", False))
    if calibration_enabled and not args.full_transformer:
        raise ValueError("calibration_mode=env_affine is only supported for FullTransformer.")
    config.debug_probe = debug_probe

    if args.full_transformer:
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
        print(f"[CONFIG] n_embd={config.n_embd}, n_gxe_layer={config.n_gxe_layer}, "
              f"n_head={config.n_head}, dropout={config.dropout}, "
              f"full_transformer={args.full_transformer}, calibration_mode={config.calibration_mode}")
        if debug_probe:
            print(
                f"[DEBUG_PROBE] enabled=True max_steps={debug_max_steps} "
                f"skip_backward={debug_skip_backward} skip_optimizer={debug_skip_optimizer} "
                f"no_autocast={debug_no_autocast}",
                flush=True,
            )
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # build loss
    loss_function = build_loss(args.loss, args.loss_weights)
    rank_aux_loss_name = _normalize_choice(
        "rank_aux_loss",
        getattr(args, "rank_aux_loss", "none"),
        {"none", "envspearman", "triplet"},
        "none",
    )
    rank_aux_loss_fn = build_loss(rank_aux_loss_name, "1.0") if rank_aux_loss_name != "none" else None
    rank_aux_weight = float(getattr(args, "rank_aux_weight", 0.05))
    envccc_loss_fn = build_loss("envccc", "1.0") if calibration_enabled else None

    # contrastive objectives (ablation mode: none, g, e, g+e)
    contrastive_mode = _normalize_choice(
        "contrastive_mode",
        getattr(args, "contrastive_mode", "none"),
        {"none", "g", "e", "g+e"},
        "none",
    )
    use_g_contrastive = contrastive_mode in {"g", "g+e"}
    use_e_contrastive = contrastive_mode in {"e", "g+e"}

    contrastive_weight = float(getattr(args, "contrastive_weight", 0.1))
    contrastive_temperature = float(getattr(args, "contrastive_temperature", 0.1))
    contrastive_sim_type = _normalize_choice(
        "contrastive_sim_type",
        getattr(args, "contrastive_sim_type", "grm"),
        {"grm", "ibs"},
        "grm",
    )
    contrastive_loss_type = _normalize_choice(
        "contrastive_loss_type",
        getattr(args, "contrastive_loss_type", "mse"),
        {"mse", "cosine", "kl"},
        "mse",
    )

    env_contrastive_weight = float(getattr(args, "env_contrastive_weight", 0.1))
    env_contrastive_temperature = float(getattr(args, "env_contrastive_temperature", 0.5))
    contrastive_warmup_epochs = int(getattr(args, "contrastive_warmup_epochs", 50))
    contrastive_ramp_epochs = int(getattr(args, "contrastive_ramp_epochs", 50))

    g_contrastive_loss_fn = None
    if use_g_contrastive:
        g_contrastive_loss_fn = GenomicContrastiveLoss(
            temperature=contrastive_temperature,
            similarity_type=contrastive_sim_type,
            loss_type=contrastive_loss_type,
        )

    e_contrastive_loss_fn = None
    if use_e_contrastive:
        e_contrastive_loss_fn = EnvironmentContrastiveLoss(
            temperature=env_contrastive_temperature
        )

    if is_main(rank):
        print(f"[INFO] Contrastive mode: {contrastive_mode}")
        print(
            f"[INFO] Rank aux: {rank_aux_loss_name}, "
            f"start={getattr(args, 'rank_aux_start_epoch', 100)}, weight={rank_aux_weight}"
        )
        if use_g_contrastive:
            print(
                f"       G contrastive: weight={contrastive_weight}, "
                f"temperature={contrastive_temperature}, "
                f"similarity_type={contrastive_sim_type}, loss_type={contrastive_loss_type}"
            )
        if use_e_contrastive:
            print(
                f"       E contrastive: weight={env_contrastive_weight}, "
                f"temperature={env_contrastive_temperature}"
            )
        if calibration_enabled:
            print(
                "[INFO] Calibration: "
                f"start={args.calibration_start_epoch}, ramp={args.calibration_ramp_epochs}, "
                f"detach_until={args.calibration_detach_rank_until_epoch}, "
                f"envccc_w={args.envccc_weight}, huber_w={args.huber_weight}"
            )

    # other options
    batches_per_epoch = len(train_loader)
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

    if val_scheme == "hybrid_combo":
        if checkpoint_metric == "select":
            selection_metric_name = "val_loss/select_score"
        elif checkpoint_metric == "proxy_scale":
            selection_metric_name = "val_loss/proxy_env_ccc"
        else:
            selection_metric_name = "val_loss/leo_env_pcc"
    else:
        selection_metric_name = "val_loss/env_avg_pearson"
    wandb_group_name = make_wandb_group_name(base_run_name)

    ### wandb logging ###
    if is_main(rank):
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=wandb_run_name,
            group=wandb_group_name,
        )
        run.config.update({
            "base_model_type": base_run_name,
            "run_name": wandb_run_name,
            "wandb_group_name": wandb_group_name,
            "selection_metric": selection_metric_name,
            "checkpoint_metric": checkpoint_metric,
            "val_scheme": val_scheme,
            "proxy_tester": proxy_tester if use_proxy_split else None,
            "proxy_holdout_frac": proxy_holdout_frac if use_proxy_split else None,
            "proxy_seed": proxy_seed if use_proxy_split else None,
            "proxy_disjoint_from_leo": bool(proxy_disjoint_from_leo) if use_proxy_split else None,
            "proxy_score_weight": args.proxy_score_weight if val_scheme == "hybrid_combo" else None,
            "leo_score_weight": args.leo_score_weight if val_scheme == "hybrid_combo" else None,
            "scale_score_weight": args.scale_score_weight if val_scheme == "hybrid_combo" else None,
            "calibration_mode": args.calibration_mode,
            "rank_aux_loss": rank_aux_loss_name,
            "rank_aux_weight": rank_aux_weight if rank_aux_loss_fn is not None else None,
            "rank_aux_start_epoch": args.rank_aux_start_epoch,
            "contrastive_warmup_epochs": contrastive_warmup_epochs,
            "contrastive_ramp_epochs": contrastive_ramp_epochs,
            "calibration_start_epoch": args.calibration_start_epoch,
            "calibration_ramp_epochs": args.calibration_ramp_epochs,
            "calibration_detach_rank_until_epoch": args.calibration_detach_rank_until_epoch,
            "calibration_joint_grad_fraction": args.calibration_joint_grad_fraction,
            "envccc_weight": args.envccc_weight if calibration_enabled else None,
            "huber_weight": args.huber_weight if calibration_enabled else None,
            "huber_delta": args.huber_delta if calibration_enabled else None,
        }, allow_val_change=True)
        run.summary["model_type"] = base_run_name
        run.summary["run_name"] = wandb_run_name
        run.summary["wandb_group_name"] = wandb_group_name
        run.summary["checkpoint_metric"] = checkpoint_metric

        # make checkpoint dir that matches run name
        run_ckpt_dir = Path("checkpoints") / wandb_run_name
        if run_ckpt_dir.exists():
            shutil.rmtree(run_ckpt_dir)
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        run.summary["checkpoint_dir"] = str(run_ckpt_dir.resolve())

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
        run.define_metric("train_loss_epoch/env_avg_pearson", step_metric="epoch")
        run.define_metric("val_loss/env_avg_pearson", step_metric="epoch")
        run.define_metric("train_loss_epoch/rank_env_pcc", step_metric="epoch")
        run.define_metric("train_loss_epoch/total_env_pcc", step_metric="epoch")
        run.define_metric("val_loss/rank_env_pcc", step_metric="epoch")
        run.define_metric("val_loss/total_env_pcc", step_metric="epoch")
        if val_scheme == "hybrid_combo":
            run.define_metric("val_loss/leo_env_pcc", step_metric="epoch")
            run.define_metric("val_loss/proxy_env_pcc", step_metric="epoch")
            run.define_metric("val_loss/proxy_env_ccc", step_metric="epoch")
            run.define_metric("val_loss/proxy_env_huber", step_metric="epoch")
            run.define_metric("val_loss/select_score", step_metric="epoch")

        # loss tracking
        wandb.config.update({"loss": args.loss,
                             "loss_weights": args.loss_weights,
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
                             "contrastive_mode": contrastive_mode,
                             "g_contrastive_enabled": use_g_contrastive,
                             "g_contrastive_weight": contrastive_weight if use_g_contrastive else None,
                             "g_contrastive_temperature": contrastive_temperature if use_g_contrastive else None,
                             "g_contrastive_sim_type": contrastive_sim_type if use_g_contrastive else None,
                             "g_contrastive_loss_type": contrastive_loss_type if use_g_contrastive else None,
                             "e_contrastive_enabled": use_e_contrastive,
                             "e_contrastive_weight": env_contrastive_weight if use_e_contrastive else None,
                             "e_contrastive_temperature": env_contrastive_temperature if use_e_contrastive else None,
                             "g_encoder_type": g_encoder_type,
                             "moe_num_experts": moe_num_experts,
                             "moe_top_k": moe_top_k,
                             "moe_expert_hidden_dim": moe_expert_hidden_dim,
                             "moe_shared_expert": moe_shared_expert,
                             "moe_shared_expert_hidden_dim": moe_shared_expert_hidden_dim,
                             "moe_loss_weight": moe_loss_weight,
                             "g_input_type": g_input_type,
                             "env_categorical_mode": env_categorical_mode,
                             "env_cat_embeddings": (env_categorical_mode == "onehot"),
                             "full_transformer": args.full_transformer,
                             "full_tf_mlp_type": full_tf_mlp_type,
                             "calibration_mode": args.calibration_mode,
                             "rank_aux_loss": rank_aux_loss_name},
                             allow_val_change=True)
        for name in loss_function.names:
            run.define_metric(f"train_loss/{name}", step_metric="iter_num")
            run.define_metric(f"train_loss_epoch/{name}", step_metric="epoch")
            run.define_metric(f"val_loss/{name}", step_metric="epoch")
        if rank_aux_loss_fn is not None:
            run.define_metric("train_loss/rank_aux", step_metric="iter_num")
            run.define_metric("train_loss/rank_aux_weight_eff", step_metric="iter_num")
        if calibration_enabled:
            run.define_metric("train_loss/calibration_envccc", step_metric="iter_num")
            run.define_metric("train_loss/calibration_huber", step_metric="iter_num")
            run.define_metric("train_loss/calibration_weight_eff_envccc", step_metric="iter_num")
            run.define_metric("train_loss/calibration_weight_eff_huber", step_metric="iter_num")
        if use_g_contrastive:
            run.define_metric("train_loss/contrastive_g", step_metric="iter_num")
            run.define_metric("train_loss_epoch/contrastive_g", step_metric="epoch")
            run.define_metric("train_loss/contrastive_weight_eff_g", step_metric="iter_num")
        if use_e_contrastive:
            run.define_metric("train_loss/contrastive_e", step_metric="iter_num")
            run.define_metric("train_loss_epoch/contrastive_e", step_metric="epoch")
            run.define_metric("train_loss/contrastive_weight_eff_e", step_metric="iter_num")
        if use_g_contrastive or use_e_contrastive:
            run.define_metric("train_loss/contrastive", step_metric="iter_num")
            run.define_metric("train_loss_epoch/contrastive", step_metric="epoch")
        if moe_encoder_enabled:
            run.define_metric("train_loss/moe_lb", step_metric="iter_num")
            run.define_metric("train_loss_epoch/moe_lb", step_metric="epoch")
            run.define_metric("val_loss/moe_lb", step_metric="epoch")
        if proxy_ds is not None:
            proxy_support_fraction = _proxy_support_fraction(train_ds, proxy_ds, proxy_tester)
            overlap_rows = 0
            if "id" in val_ds.meta.columns and "id" in proxy_ds.meta.columns:
                overlap_rows = len(set(val_ds.meta["id"].astype(str)) & set(proxy_ds.meta["id"].astype(str)))
            run.summary["proxy/row_count"] = int(len(proxy_ds))
            run.summary["proxy/hybrid_count"] = int(proxy_ds.meta['Hybrid'].astype(str).nunique())
            run.summary["proxy/env_count"] = int(proxy_ds.meta['Env'].astype(str).nunique())
            run.summary["proxy/year_distribution"] = json.dumps(
                {str(k): int(v) for k, v in proxy_ds.meta['Year'].value_counts().sort_index().to_dict().items()},
                sort_keys=True,
            )
            run.summary["proxy/support_fraction"] = float(proxy_support_fraction)
            run.summary["proxy/leo_overlap_rows"] = int(overlap_rows)

    # initialize training states
    best_val_loss = float("inf")
    best_val_env_pcc = -float("inf")
    best_select_score = -float("inf")
    best_proxy_huber = float("inf")
    best_select_val_loss = float("inf")
    best_scale_score = -float("inf")
    best_scale_huber = float("inf")
    best_scale_val_loss = float("inf")
    last_improved = 0
    iter_num = 0
    t0 = time.time()
    checkpoint_alias_state = {}
    manifest_path = None
    if is_main(rank):
        manifest_path = run_ckpt_dir / "checkpoint_manifest.json"
    if is_main(rank):
        print(f"[INFO] Checkpoint/early-stop selection metric: {selection_metric_name}")

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

            if debug_probe and step_idx == 0 and is_main(rank):
                g_cpu = xb.get("g_data")
                e_cpu = xb.get("e_data")
                if g_cpu is not None:
                    g_min = int(g_cpu.min().item())
                    g_max = int(g_cpu.max().item())
                    uniq = torch.unique(g_cpu.view(-1))
                    uniq_preview = uniq[:16].tolist()
                    print(
                        f"[DEBUG_PROBE] cpu_batch/g_data shape={tuple(g_cpu.shape)} "
                        f"dtype={g_cpu.dtype} min={g_min} max={g_max} "
                        f"unique_preview={uniq_preview}",
                        flush=True,
                    )
                    if args.full_transformer and getattr(model.module, "g_input_type", "tokens") == "tokens":
                        vocab_size = int(model.module.g_embed.num_embeddings)
                        print(f"[DEBUG_PROBE] cpu_batch/g_vocab_size={vocab_size}", flush=True)
                        if g_min < 0 or g_max >= vocab_size:
                            raise ValueError(
                                f"Token index out of range for embedding: min={g_min}, max={g_max}, vocab_size={vocab_size}"
                            )
                if e_cpu is not None:
                    print(
                        f"[DEBUG_PROBE] cpu_batch/e_data shape={tuple(e_cpu.shape)} "
                        f"dtype={e_cpu.dtype} finite={bool(torch.isfinite(e_cpu).all().item())}",
                        flush=True,
                    )

            for k, v in xb.items():
                xb[k] = v.to(device, non_blocking=True)
            y_true = yb["y"].to(device, non_blocking=True).float()
            env_id = yb["env_id"].to(device, non_blocking=True).long()
            if debug_probe and step_idx == 0:
                _debug_sync_probe(debug_probe, rank, "batch_on_device/y_true", y_true)
                _debug_sync_probe(debug_probe, rank, "batch_on_device/env_id", env_id.float())

            contrastive_factor = _ramp_factor(epoch_num, contrastive_warmup_epochs, contrastive_ramp_epochs)
            calibration_factor = (
                _ramp_factor(epoch_num, args.calibration_start_epoch, args.calibration_ramp_epochs)
                if calibration_enabled else 0.0
            )
            detach_rank_in_calibration = calibration_enabled and (epoch_num < args.calibration_detach_rank_until_epoch)
            calibration_rank_grad_scale = 1.0
            if calibration_enabled and not detach_rank_in_calibration:
                calibration_rank_grad_scale = float(args.calibration_joint_grad_fraction)

            # fwd/bwd pass
            autocast_ctx = (
                nullcontext()
                if debug_no_autocast
                else torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            )
            _debug_sync_probe(debug_probe and step_idx == 0, rank, "before_forward")
            with autocast_ctx:
                if use_g_contrastive and use_e_contrastive:
                    if calibration_enabled:
                        preds, g_embeddings, e_embeddings = model(
                            xb,
                            return_g_embeddings=True,
                            return_e_embeddings=True,
                            detach_rank_in_calibration=detach_rank_in_calibration,
                            calibration_rank_grad_scale=calibration_rank_grad_scale,
                        )
                    else:
                        preds, g_embeddings, e_embeddings = model(
                            xb,
                            return_g_embeddings=True,
                            return_e_embeddings=True,
                        )
                elif use_g_contrastive:
                    if calibration_enabled:
                        preds, g_embeddings = model(
                            xb,
                            return_g_embeddings=True,
                            detach_rank_in_calibration=detach_rank_in_calibration,
                            calibration_rank_grad_scale=calibration_rank_grad_scale,
                        )
                    else:
                        preds, g_embeddings = model(xb, return_g_embeddings=True)
                    e_embeddings = None
                elif use_e_contrastive:
                    if calibration_enabled:
                        preds, e_embeddings = model(
                            xb,
                            return_e_embeddings=True,
                            detach_rank_in_calibration=detach_rank_in_calibration,
                            calibration_rank_grad_scale=calibration_rank_grad_scale,
                        )
                    else:
                        preds, e_embeddings = model(xb, return_e_embeddings=True)
                    g_embeddings = None
                else:
                    if calibration_enabled:
                        preds = model(
                            xb,
                            detach_rank_in_calibration=detach_rank_in_calibration,
                            calibration_rank_grad_scale=calibration_rank_grad_scale,
                        )
                    else:
                        preds = model(xb)
                    g_embeddings = None
                    e_embeddings = None

                _debug_sync_probe(debug_probe and step_idx == 0, rank, "after_forward")
                total_pred, rank_pred = _extract_prediction_tensors(preds)
                if debug_probe and step_idx == 0:
                    _debug_sync_probe(True, rank, "rank_pred", rank_pred)
                    _debug_sync_probe(True, rank, "total_pred", total_pred)
                loss_total, loss_parts = loss_function(rank_pred, y_true, env_id=env_id)

                if rank_aux_loss_fn is not None and epoch_num >= int(args.rank_aux_start_epoch):
                    rank_aux_total, _ = rank_aux_loss_fn(rank_pred, y_true, env_id=env_id)
                    loss_total = loss_total + (rank_aux_weight * rank_aux_total)
                    loss_parts["rank_aux"] = float(rank_aux_total.detach().item())
                    loss_parts["rank_aux_weight_eff"] = rank_aux_weight
                elif rank_aux_loss_fn is not None:
                    loss_parts["rank_aux"] = 0.0
                    loss_parts["rank_aux_weight_eff"] = 0.0

                if use_g_contrastive or use_e_contrastive:
                    if contrastive_factor > 0.0:
                        contrastive_total = 0.0

                        if use_g_contrastive and g_embeddings is not None and g_contrastive_loss_fn is not None:
                            g_contr = g_contrastive_loss_fn(g_embeddings, g_data=xb["g_data"])
                            g_weight_eff = contrastive_weight * contrastive_factor
                            loss_total = loss_total + g_weight_eff * g_contr
                            loss_parts["contrastive_g"] = float(g_contr.detach().item())
                            loss_parts["contrastive_weight_eff_g"] = g_weight_eff
                            contrastive_total += float(g_contr.detach().item())
                        elif use_g_contrastive:
                            loss_parts["contrastive_g"] = 0.0
                            loss_parts["contrastive_weight_eff_g"] = 0.0

                        if use_e_contrastive and e_embeddings is not None and e_contrastive_loss_fn is not None:
                            e_contr = e_contrastive_loss_fn(
                                e_embeddings,
                                e_data=xb["e_data"],
                            )
                            e_weight_eff = env_contrastive_weight * contrastive_factor
                            loss_total = loss_total + e_weight_eff * e_contr
                            loss_parts["contrastive_e"] = float(e_contr.detach().item())
                            loss_parts["contrastive_weight_eff_e"] = e_weight_eff
                            contrastive_total += float(e_contr.detach().item())
                        elif use_e_contrastive:
                            loss_parts["contrastive_e"] = 0.0
                            loss_parts["contrastive_weight_eff_e"] = 0.0

                        loss_parts["contrastive"] = contrastive_total
                    else:
                        if use_g_contrastive:
                            loss_parts["contrastive_g"] = 0.0
                            loss_parts["contrastive_weight_eff_g"] = 0.0
                        if use_e_contrastive:
                            loss_parts["contrastive_e"] = 0.0
                            loss_parts["contrastive_weight_eff_e"] = 0.0
                        loss_parts["contrastive"] = 0.0

                if calibration_enabled:
                    if calibration_factor > 0.0:
                        envccc_total, _ = envccc_loss_fn(total_pred, y_true, env_id=env_id)
                        huber_total = F.huber_loss(
                            total_pred.float(),
                            y_true.float(),
                            reduction="mean",
                            delta=float(args.huber_delta),
                        )
                        eff_envccc = float(args.envccc_weight) * calibration_factor
                        eff_huber = float(args.huber_weight) * calibration_factor
                        loss_total = loss_total + (eff_envccc * envccc_total) + (eff_huber * huber_total)
                        loss_parts["calibration_envccc"] = float(envccc_total.detach().item())
                        loss_parts["calibration_huber"] = float(huber_total.detach().item())
                        loss_parts["calibration_weight_eff_envccc"] = eff_envccc
                        loss_parts["calibration_weight_eff_huber"] = eff_huber
                    else:
                        loss_parts["calibration_envccc"] = 0.0
                        loss_parts["calibration_huber"] = 0.0
                        loss_parts["calibration_weight_eff_envccc"] = 0.0
                        loss_parts["calibration_weight_eff_huber"] = 0.0

                moe_aux_loss = getattr(model.module, "moe_aux_loss", None)
                if moe_aux_loss is not None:
                    # Don't detach - gradients need to flow to gate network for load balancing
                    # DDP handles this with find_unused_parameters=True
                    loss_total = loss_total + moe_aux_loss
                    loss_parts["moe_lb"] = float(moe_aux_loss.detach().item())
            _debug_sync_probe(debug_probe and step_idx == 0, rank, "after_loss")
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
            if not debug_skip_backward:
                loss_total.backward()
                _debug_sync_probe(debug_probe and step_idx == 0, rank, "after_backward")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            else:
                if debug_probe and step_idx == 0 and is_main(rank):
                    print("[DEBUG_PROBE] skipping backward()", flush=True)
            if not debug_skip_optimizer:
                optimizer.step()
                _debug_sync_probe(debug_probe and step_idx == 0, rank, "after_optimizer")
            else:
                if debug_probe and step_idx == 0 and is_main(rank):
                    print("[DEBUG_PROBE] skipping optimizer.step()", flush=True)
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

            # Timing diagnostics
            if epoch_num == 0 and is_main(rank):
                step_times.append(time.time() - t_step_start)
                if step_idx < 5 or step_idx % 20 == 0:
                    print(f"[TIMING] Step {step_idx}: {step_times[-1]:.3f}s (avg: {sum(step_times)/len(step_times):.3f}s)")

            iter_num += 1
            pbar.update(1)
            if debug_max_steps > 0 and (step_idx + 1) >= debug_max_steps:
                if is_main(rank):
                    print(f"[DEBUG_PROBE] stopping after {debug_max_steps} step(s)", flush=True)
                break
        pbar.close()

        # End of epoch timing
        if epoch_num == 0 and is_main(rank):
            t_epoch_end = time.time()
            print(f"[TIMING] Epoch 0: {t_epoch_end - t_epoch_start:.1f}s total, {len(step_times)} steps, {sum(step_times)/len(step_times):.3f}s/step avg")

        ### evaluation ###
        with torch.no_grad():
            model.eval()

            def _all_gather_flat(t):
                local_n = torch.tensor([t.shape[0]], device=device)
                all_n = [torch.zeros_like(local_n) for _ in range(world_size)]
                dist.all_gather(all_n, local_n)
                max_n = max(x.item() for x in all_n)
                padded = torch.zeros(max_n, device=device, dtype=t.dtype)
                padded[:t.shape[0]] = t
                gathered = [torch.zeros_like(padded) for _ in range(world_size)]
                dist.all_gather(gathered, padded)
                parts = [gathered[i][:all_n[i].item()] for i in range(world_size)]
                return torch.cat(parts)

            def _gather_predictions(loader, dataset_len, max_batches=None):
                """Gather ALL predictions/targets across batches and DDP ranks,
                then compute metrics on the full dataset.  This aligns the
                validation metric with how eval.py computes test metrics."""
                all_total_preds = []
                all_rank_preds = []
                all_targets = []
                all_env_ids = []
                all_scales = []
                all_shifts = []

                for i, (xb, yb) in enumerate(loader):
                    if (max_batches is not None) and (i >= max_batches):
                        break
                    xb = {k: v.to(device, non_blocking=True) for k, v in xb.items()}
                    y_true = yb["y"].to(device, non_blocking=True).float()
                    env_id = yb["env_id"].to(device, non_blocking=True).long()

                    preds = model(xb)
                    total_pred, rank_pred = _extract_prediction_tensors(preds)
                    all_total_preds.append(total_pred.squeeze(-1))
                    all_rank_preds.append(rank_pred.squeeze(-1))
                    all_targets.append(y_true.squeeze(-1))
                    all_env_ids.append(env_id)
                    if isinstance(preds, dict):
                        if preds.get("scale") is not None:
                            all_scales.append(preds["scale"].squeeze(-1))
                        if preds.get("shift") is not None:
                            all_shifts.append(preds["shift"].squeeze(-1))

                local_total_preds = torch.cat(all_total_preds) if all_total_preds else torch.empty(0, device=device)
                local_rank_preds = torch.cat(all_rank_preds) if all_rank_preds else torch.empty(0, device=device)
                local_targets = torch.cat(all_targets) if all_targets else torch.empty(0, device=device)
                local_env_ids = torch.cat(all_env_ids) if all_env_ids else torch.empty(0, dtype=torch.long, device=device)
                local_scales = torch.cat(all_scales) if all_scales else None
                local_shifts = torch.cat(all_shifts) if all_shifts else None

                full_total_preds = _all_gather_flat(local_total_preds)[:dataset_len]
                full_rank_preds = _all_gather_flat(local_rank_preds)[:dataset_len]
                full_targets = _all_gather_flat(local_targets)[:dataset_len]
                full_env_ids = _all_gather_flat(local_env_ids.float()).long()[:dataset_len]
                full_scales = _all_gather_flat(local_scales)[:dataset_len] if local_scales is not None else None
                full_shifts = _all_gather_flat(local_shifts)[:dataset_len] if local_shifts is not None else None
                return full_total_preds, full_rank_preds, full_targets, full_env_ids, full_scales, full_shifts

            def _max_within_env_std(values, env_ids):
                if values is None or values.numel() == 0:
                    return 0.0
                max_std = 0.0
                for env_value in torch.unique(env_ids):
                    env_vals = values[env_ids == env_value]
                    if env_vals.numel() > 1:
                        env_std = float(env_vals.float().std(unbiased=False).item())
                        max_std = max(max_std, env_std)
                return max_std

            def eval_loader(loader, dataset_len, max_batches=None):
                """Compute validation metrics using gather-then-compute approach."""
                full_total_preds, full_rank_preds, full_targets, full_env_ids, full_scales, full_shifts = \
                    _gather_predictions(loader, dataset_len, max_batches)

                ltot, lparts = loss_function(full_rank_preds, full_targets, env_id=full_env_ids)
                rank_env_pcc = macro_env_pearson(
                    full_rank_preds, full_targets, full_env_ids, min_samples=2
                )
                total_env_pcc = macro_env_pearson(
                    full_total_preds, full_targets, full_env_ids, min_samples=2
                )
                env_ccc = macro_env_ccc(
                    full_total_preds, full_targets, full_env_ids, min_samples=2
                )
                huber = F.huber_loss(
                    full_total_preds.float(),
                    full_targets.float(),
                    reduction="mean",
                    delta=float(args.huber_delta),
                )
                scale_env_std_max = _max_within_env_std(full_scales, full_env_ids)
                shift_env_std_max = _max_within_env_std(full_shifts, full_env_ids)
                rank_env_pcc_value = float(rank_env_pcc.item()) if torch.isfinite(rank_env_pcc) else float("nan")
                total_env_pcc_value = float(total_env_pcc.item()) if torch.isfinite(total_env_pcc) else float("nan")
                if calibration_enabled and math.isfinite(rank_env_pcc_value) and math.isfinite(total_env_pcc_value):
                    if abs(rank_env_pcc_value - total_env_pcc_value) > 1e-5:
                        raise RuntimeError(
                            "rank_env_pcc and total_env_pcc diverged under env_affine calibration: "
                            f"{rank_env_pcc_value:.8f} vs {total_env_pcc_value:.8f}"
                        )
                if calibration_enabled and max(scale_env_std_max, shift_env_std_max) > 1e-5:
                    raise RuntimeError(
                        "env_affine calibration produced per-environment-varying scale/shift: "
                        f"max_scale_std={scale_env_std_max:.8e}, max_shift_std={shift_env_std_max:.8e}"
                    )
                return {
                    "loss_total": ltot,
                    "loss_parts": {k: float(v) for k, v in lparts.items()},
                    "rank_env_pcc": rank_env_pcc_value,
                    "total_env_pcc": total_env_pcc_value,
                    "env_pcc": rank_env_pcc_value,
                    "env_ccc": float(env_ccc.item()) if torch.isfinite(env_ccc) else float("nan"),
                    "huber": float(huber.item()),
                    "scale_env_std_max": scale_env_std_max,
                    "shift_env_std_max": shift_env_std_max,
                }

            # reshuffle train sampler s.t. sampled subset is different each epoch
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(10_000 + epoch_num)
            # eval on subset of train for speed, full val
            train_metrics = eval_loader(
                train_loader,
                len(train_ds),
                max_batches=int(math.ceil(len(val_loader) / world_size)),
            )
            val_metrics = eval_loader(
                val_loader,
                len(val_ds),
                max_batches=None,
            )
            proxy_metrics = None
            if proxy_loader is not None:
                proxy_metrics = eval_loader(
                    proxy_loader,
                    len(proxy_ds),
                    max_batches=None,
                )

        # log eval / early stop (only rank 0)
        if is_main(rank):
            train_total = train_metrics["loss_total"]
            train_parts = train_metrics["loss_parts"]
            train_rank_env_pcc = train_metrics["rank_env_pcc"]
            train_total_env_pcc = train_metrics["total_env_pcc"]
            val_total = val_metrics["loss_total"]
            val_parts = val_metrics["loss_parts"]
            val_rank_env_pcc = val_metrics["rank_env_pcc"]
            val_total_env_pcc = val_metrics["total_env_pcc"]
            log_epoch_payload = {
                "epoch": epoch_num,
                "val_loss": val_total.item(),
                "val_loss/env_avg_pearson": val_rank_env_pcc,
                "val_loss/rank_env_pcc": val_rank_env_pcc,
                "val_loss/total_env_pcc": val_total_env_pcc,
            }
            for k, v in val_parts.items():
                log_epoch_payload[f"val_loss/{k}"] = float(v)
            log_epoch_payload["train_loss_epoch"] = train_total.item()
            log_epoch_payload["train_loss_epoch/env_avg_pearson"] = train_rank_env_pcc
            log_epoch_payload["train_loss_epoch/rank_env_pcc"] = train_rank_env_pcc
            log_epoch_payload["train_loss_epoch/total_env_pcc"] = train_total_env_pcc
            for k, v in train_parts.items():
                log_epoch_payload[f"train_loss_epoch/{k}"] = float(v)
            select_score = None
            proxy_env_huber = float("inf")
            proxy_env_ccc_value = -float("inf")
            if val_scheme == "hybrid_combo":
                log_epoch_payload["val_loss/leo_env_pcc"] = val_rank_env_pcc
                if proxy_metrics is not None:
                    log_epoch_payload["val_loss/proxy_env_pcc"] = proxy_metrics["rank_env_pcc"]
                    log_epoch_payload["val_loss/proxy_env_ccc"] = proxy_metrics["env_ccc"]
                    log_epoch_payload["val_loss/proxy_env_huber"] = proxy_metrics["huber"]
                    proxy_env_pcc_score = proxy_metrics["rank_env_pcc"] if math.isfinite(proxy_metrics["rank_env_pcc"]) else -1.0
                    proxy_env_ccc_score = proxy_metrics["env_ccc"] if math.isfinite(proxy_metrics["env_ccc"]) else -1.0
                    proxy_env_ccc_value = proxy_metrics["env_ccc"] if math.isfinite(proxy_metrics["env_ccc"]) else -float("inf")
                    leo_env_pcc_score = val_rank_env_pcc if math.isfinite(val_rank_env_pcc) else -1.0
                    select_score = (
                        float(args.proxy_score_weight) * proxy_env_pcc_score
                        + float(args.leo_score_weight) * leo_env_pcc_score
                        + float(args.scale_score_weight) * proxy_env_ccc_score
                    )
                    log_epoch_payload["val_loss/select_score"] = select_score
                    proxy_env_huber = proxy_metrics["huber"]
            wandb.log(log_epoch_payload)

            val_loss_value = float(val_total.item())
            leo_improved = _metric_improved(
                val_rank_env_pcc,
                best_val_env_pcc,
                [(val_loss_value, best_val_loss)],
            )
            select_improved = select_score is not None and _metric_improved(
                select_score,
                best_select_score,
                [(proxy_env_huber, best_proxy_huber), (val_loss_value, best_select_val_loss)],
            )
            scale_improved = proxy_metrics is not None and _metric_improved(
                proxy_env_ccc_value,
                best_scale_score,
                [(proxy_env_huber, best_scale_huber), (val_loss_value, best_scale_val_loss)],
            )

            if val_scheme == "hybrid_combo":
                if checkpoint_metric == "select":
                    canonical_improved = bool(select_improved)
                elif checkpoint_metric == "proxy_scale":
                    canonical_improved = bool(scale_improved)
                else:
                    canonical_improved = bool(leo_improved)
            else:
                canonical_improved = bool(leo_improved)

            if leo_improved:
                best_val_loss = val_loss_value
                best_val_env_pcc = val_rank_env_pcc
            if select_improved:
                best_select_score = float(select_score)
                best_proxy_huber = proxy_env_huber
                best_select_val_loss = val_loss_value
            if scale_improved:
                best_scale_score = proxy_env_ccc_value
                best_scale_huber = proxy_env_huber
                best_scale_val_loss = val_loss_value

            if canonical_improved:
                last_improved = 0
            else:
                last_improved += 1
                print(f"Validation has not improved in {last_improved} steps")

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
            marker_stats_payload = None
            if getattr(train_ds, "marker_stats", None):
                marker_stats_payload = {
                    "p": train_ds.marker_stats["p"].tolist(),
                    "scale": train_ds.marker_stats["scale"].tolist(),
                    "valid": train_ds.marker_stats["valid"].tolist(),
                    "columns": list(train_ds.marker_stats["columns"]),
                }

            metric_snapshot = _safe_metric_snapshot(
                selection_metric_name=selection_metric_name,
                checkpoint_metric=checkpoint_metric,
                val_loss=val_loss_value,
                val_rank_env_pcc=val_rank_env_pcc,
                val_total_env_pcc=val_total_env_pcc,
                proxy_env_pcc=(proxy_metrics["rank_env_pcc"] if proxy_metrics is not None else None),
                proxy_env_ccc=(proxy_metrics["env_ccc"] if proxy_metrics is not None else None),
                proxy_env_huber=(proxy_metrics["huber"] if proxy_metrics is not None else None),
                select_score=select_score,
            )

            ckpt = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_num,
                "val_loss": val_loss_value,
                "val_env_avg_pearson": val_rank_env_pcc,
                "checkpoint_metric": checkpoint_metric,
                "checkpoint_metrics": metric_snapshot,
                "config": {
                    "g_enc": args.g_enc,
                    "e_enc": args.e_enc,
                    "ld_enc": args.ld_enc,
                    "gxe_enc": args.gxe_enc,
                    "g_encoder_type": g_encoder_type,
                    "full_transformer": args.full_transformer,
                    "full_tf_mlp_type": full_tf_mlp_type,
                    "block_size": config.block_size,
                    "n_env_fts": config.n_env_fts,
                    "g_layers": args.g_layers,
                    "ld_layers": args.ld_layers,
                    "mlp_layers": args.mlp_layers,
                    "gxe_layers": args.gxe_layers,
                    "n_head": args.heads,
                    "n_embd": args.emb_size,
                    "moe_num_experts": moe_num_experts,
                    "moe_top_k": moe_top_k,
                    "moe_expert_hidden_dim": moe_expert_hidden_dim,
                    "moe_shared_expert": moe_shared_expert,
                    "moe_shared_expert_hidden_dim": moe_shared_expert_hidden_dim,
                    "moe_loss_weight": moe_loss_weight,
                    "g_input_type": g_input_type,
                    "env_categorical_mode": env_categorical_mode,
                    "env_cat_embeddings": (env_categorical_mode == "onehot"),
                    "loss": args.loss,
                    "loss_weights": args.loss_weights,
                    "scale_targets": args.scale_targets,
                    "calibration_mode": args.calibration_mode,
                    "rank_aux_loss": rank_aux_loss_name,
                    "rank_aux_weight": rank_aux_weight if rank_aux_loss_fn is not None else None,
                    "rank_aux_start_epoch": args.rank_aux_start_epoch,
                    "contrastive_warmup_epochs": contrastive_warmup_epochs,
                    "contrastive_ramp_epochs": contrastive_ramp_epochs,
                    "val_scheme": val_scheme,
                    "checkpoint_metric": checkpoint_metric,
                    "proxy_tester": proxy_tester if use_proxy_split else None,
                    "proxy_holdout_frac": proxy_holdout_frac if use_proxy_split else None,
                    "proxy_seed": proxy_seed if use_proxy_split else None,
                    "proxy_disjoint_from_leo": bool(proxy_disjoint_from_leo) if use_proxy_split else None,
                    "proxy_heldout_parent1": proxy_split_info.get("heldout_parent1") if proxy_split_info else None,
                    "proxy_score_weight": float(args.proxy_score_weight) if proxy_metrics is not None else None,
                    "leo_score_weight": float(args.leo_score_weight) if proxy_metrics is not None else None,
                    "scale_score_weight": float(args.scale_score_weight) if proxy_metrics is not None else None,
                    "calibration_start_epoch": args.calibration_start_epoch,
                    "calibration_ramp_epochs": args.calibration_ramp_epochs,
                    "calibration_detach_rank_until_epoch": args.calibration_detach_rank_until_epoch,
                    "calibration_joint_grad_fraction": args.calibration_joint_grad_fraction,
                    "envccc_weight": args.envccc_weight if calibration_enabled else None,
                    "huber_weight": args.huber_weight if calibration_enabled else None,
                    "huber_delta": args.huber_delta if calibration_enabled else None,
                    "saved_calibration_detach_rank": bool(detach_rank_in_calibration),
                },
                "env_scaler": env_scaler_payload,
                "y_scalers": label_scalers_payload,
                "marker_stats": marker_stats_payload,
                "run": {"id": run.id if 'run' in locals() else None,
                        "name": wandb_run_name}
            }

            latest_path = run_ckpt_dir / "latest.pt"
            torch.save(ckpt, latest_path)
            _update_checkpoint_manifest(
                manifest_path,
                checkpoint_alias_state,
                "latest",
                latest_path,
                epoch_num,
                metric_snapshot,
            )

            if canonical_improved:
                ckpt_path = run_ckpt_dir / f"checkpoint_{epoch_num:04d}.pt"
                shutil.copy2(latest_path, ckpt_path)

            if leo_improved:
                alias_path = run_ckpt_dir / "best_leo.pt"
                shutil.copy2(latest_path, alias_path)
                _update_checkpoint_manifest(
                    manifest_path,
                    checkpoint_alias_state,
                    "best_leo",
                    alias_path,
                    epoch_num,
                    metric_snapshot,
                )
                print(
                    "*** best_leo updated: "
                    f"leo_env_pcc={best_val_env_pcc:.5f} "
                    f"(val_loss={best_val_loss:.4e}) ***"
                )
            if select_improved:
                alias_path = run_ckpt_dir / "best_select.pt"
                shutil.copy2(latest_path, alias_path)
                _update_checkpoint_manifest(
                    manifest_path,
                    checkpoint_alias_state,
                    "best_select",
                    alias_path,
                    epoch_num,
                    metric_snapshot,
                )
                print(
                    "*** best_select updated: "
                    f"select_score={best_select_score:.5f} "
                    f"(leo_env_pcc={val_rank_env_pcc:.5f}, "
                    f"proxy_env_pcc={proxy_metrics['rank_env_pcc']:.5f}, "
                    f"proxy_env_ccc={proxy_metrics['env_ccc']:.5f}, "
                    f"proxy_env_huber={proxy_metrics['huber']:.5f}) ***"
                )
            if scale_improved:
                alias_path = run_ckpt_dir / "best_scale.pt"
                shutil.copy2(latest_path, alias_path)
                _update_checkpoint_manifest(
                    manifest_path,
                    checkpoint_alias_state,
                    "best_scale",
                    alias_path,
                    epoch_num,
                    metric_snapshot,
                )
                print(
                    "*** best_scale updated: "
                    f"proxy_env_ccc={best_scale_score:.5f} "
                    f"(proxy_env_huber={best_scale_huber:.5f}, val_loss={best_scale_val_loss:.4e}) ***"
                )

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
            if val_scheme == "hybrid_combo" and proxy_metrics is not None and select_score is not None:
                print(
                    f"[Epoch {epoch_num}] train_loss={train_total.item():.4e} | "
                    f"val_loss={val_total.item():.4e} | "
                    f"leo_env_pcc={val_rank_env_pcc:.5f} | "
                    f"proxy_env_pcc={proxy_metrics['rank_env_pcc']:.5f} | "
                    f"proxy_env_ccc={proxy_metrics['env_ccc']:.5f} | "
                    f"select_score={select_score:.5f} | "
                    f"best_leo={best_val_env_pcc:.5f} | "
                    f"best_select_score={best_select_score:.5f} | "
                    f"best_scale_ccc={best_scale_score:.5f} | "
                    f"checkpoint_metric={checkpoint_metric} | "
                    f"elapsed={elapsed:.2f}m"
                )
            else:
                print(
                    f"[Epoch {epoch_num}] train_loss={train_total.item():.4e} | "
                    f"val_loss={val_total.item():.4e} | "
                    f"rank_env_pcc={val_rank_env_pcc:.5f} | "
                    f"total_env_pcc={val_total_env_pcc:.5f} | "
                    f"best_env_pcc={best_val_env_pcc:.5f} | "
                    f"elapsed={elapsed:.2f}m"
                )

    dist.barrier()
    if is_main(rank):
        run.finish()
    cleanup_ddp()

if __name__ == "__main__":
    main()
