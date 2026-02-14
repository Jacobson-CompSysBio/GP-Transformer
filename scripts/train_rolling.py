# imports
import time, os, random, math, sys
import shutil
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import subprocess
import numpy as np
import wandb
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler

# add parent directory (one level up) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.dataset import *
from models.model import *
from models.config import Config
from utils.get_lr import get_lr
from utils.loss import (
    build_loss,
    GenomicContrastiveLoss,
    EnvironmentContrastiveLoss,
    macro_env_pearson,
)
from utils.utils import *
from utils.utils import EnvStratifiedSampler, str2bool

load_dotenv()
os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")


def extract_master_addr():
    try:
        nodelist = os.environ["SLURM_NODELIST"]
        node = subprocess.check_output(["scontrol", "show", "hostname", nodelist]).decode().splitlines()[0]
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

    master = os.environ.get("MASTER_ADDR", "")
    if not master or ":" in master:
        master = extract_master_addr()
        os.environ["MASTER_ADDR"] = master
    if rank == 0:
        print(f"[DDP] MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}, world_size={world_size}")

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    return torch.device(f"cuda:{local_rank}"), local_rank, rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank) -> bool:
    return rank == 0


def _normalize_choice(name: str, value, allowed: set[str], default: str) -> str:
    v = str(value).strip().lower()
    if v in {"", "false", "0", "off", "none", "no"}:
        return default
    if v not in allowed:
        raise ValueError(f"Unsupported {name}='{value}'. Allowed: {sorted(allowed)}")
    return v


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _build_rolling_folds() -> list[tuple[int, int]]:
    """
    Default: train<=2014..2022, val=2015..2023.
    Optional overrides via env:
      - ROLLING_VAL_YEARS="2021,2022,2023"
      - ROLLING_TRAIN_START_YEAR=2014
      - ROLLING_TRAIN_END_YEAR=2023
      - ROLLING_MAX_FOLDS=3
      - ROLLING_RECENT_FIRST=True
    """
    val_years_raw = os.getenv("ROLLING_VAL_YEARS", "").strip()
    if val_years_raw:
        val_years = sorted({int(x.strip()) for x in val_years_raw.split(",") if x.strip()})
        folds = [(y - 1, y) for y in val_years]
    else:
        start_year = int(os.getenv("ROLLING_TRAIN_START_YEAR", "2014"))
        end_year = int(os.getenv("ROLLING_TRAIN_END_YEAR", "2023"))
        if end_year <= start_year:
            raise ValueError("ROLLING_TRAIN_END_YEAR must be > ROLLING_TRAIN_START_YEAR")
        folds = [(t, t + 1) for t in range(start_year, end_year)]

    if str2bool(os.getenv("ROLLING_RECENT_FIRST", "False")):
        folds = list(reversed(folds))

    max_folds_raw = os.getenv("ROLLING_MAX_FOLDS", "").strip()
    if max_folds_raw:
        max_folds = max(1, int(max_folds_raw))
        folds = folds[:max_folds]

    return folds


def _safe_pcc_np(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rebuild_env_scaler(payload: dict | None) -> StandardScaler | None:
    if not payload:
        return None
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["mean"], dtype=float)
    scaler.scale_ = np.asarray(payload["scale"], dtype=float)
    scaler.var_ = np.asarray(payload["var"], dtype=float)
    scaler.n_features_in_ = int(payload["n_features_in"])
    if "feature_names_in" in payload:
        scaler.feature_names_in_ = np.asarray(payload["feature_names_in"], dtype=object)
    return scaler


def _rebuild_y_scalers(payload: dict | None) -> dict | None:
    if not payload:
        return None
    return {k: LabelScaler(v["mean"], v["std"]) for k, v in payload.items()}


def _rebuild_marker_stats(payload: dict | None) -> dict | None:
    if not payload:
        return None
    return {
        "p": np.asarray(payload["p"], dtype=np.float32),
        "scale": np.asarray(payload["scale"], dtype=np.float32),
        "valid": np.asarray(payload["valid"], dtype=np.float32),
        "columns": list(payload["columns"]),
    }


def _build_model_from_checkpoint_payload(
    payload: dict,
    args,
    device: torch.device,
):
    cfg = payload.get("config", {})

    g_enc = cfg.get("g_enc", args.g_enc)
    e_enc = cfg.get("e_enc", args.e_enc)
    ld_enc = cfg.get("ld_enc", args.ld_enc)
    gxe_enc = cfg.get("gxe_enc", args.gxe_enc)
    residual = bool(cfg.get("residual", args.residual))
    full_transformer = bool(cfg.get("full_transformer", args.full_transformer))
    g_encoder_type = cfg.get("g_encoder_type", getattr(args, "g_encoder_type", "dense"))
    g_input_type = cfg.get("g_input_type", getattr(args, "g_input_type", "tokens"))
    full_tf_mlp_type = cfg.get("full_tf_mlp_type", g_encoder_type)
    if isinstance(full_tf_mlp_type, str):
        full_tf_mlp_type = full_tf_mlp_type.lower()
    else:
        full_tf_mlp_type = "moe" if full_tf_mlp_type else "dense"

    moe_num_experts = int(cfg.get("moe_num_experts", getattr(args, "moe_num_experts", 4)))
    moe_top_k = int(cfg.get("moe_top_k", getattr(args, "moe_top_k", 2)))
    moe_expert_hidden_dim = cfg.get("moe_expert_hidden_dim", getattr(args, "moe_expert_hidden_dim", None))
    moe_shared_expert = bool(cfg.get("moe_shared_expert", getattr(args, "moe_shared_expert", False)))
    moe_shared_expert_hidden_dim = cfg.get(
        "moe_shared_expert_hidden_dim",
        getattr(args, "moe_shared_expert_hidden_dim", None),
    )
    moe_loss_weight = float(cfg.get("moe_loss_weight", getattr(args, "moe_loss_weight", 0.01)))

    config = Config(
        block_size=int(cfg["block_size"]),
        g_input_type=str(g_input_type).lower(),
        n_head=int(cfg.get("n_head", args.heads)),
        n_g_layer=int(cfg.get("g_layers", args.g_layers)),
        n_ld_layer=int(cfg.get("ld_layers", args.ld_layers)),
        n_mlp_layer=int(cfg.get("mlp_layers", args.mlp_layers)),
        n_gxe_layer=int(cfg.get("gxe_layers", args.gxe_layers)),
        n_embd=int(cfg.get("n_embd", args.emb_size)),
        dropout=float(cfg.get("dropout", args.dropout)),
        n_env_fts=int(cfg["n_env_fts"]),
    )

    if full_transformer:
        if residual:
            model = FullTransformerResidual(
                config,
                mlp_type=full_tf_mlp_type,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                moe_expert_hidden_dim=moe_expert_hidden_dim,
                moe_shared_expert=moe_shared_expert,
                moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
                moe_loss_weight=moe_loss_weight,
                residual=residual,
            ).to(device)
            model.detach_ymean_in_sum = bool(cfg.get("detach_ymean", getattr(args, "detach_ymean", True)))
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
    elif residual:
        model = GxE_ResidualTransformer(
            g_enc=g_enc,
            e_enc=e_enc,
            ld_enc=ld_enc,
            gxe_enc=gxe_enc,
            moe=cfg.get("wg", cfg.get("moe", args.wg)),
            g_encoder_type=g_encoder_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_expert_hidden_dim=moe_expert_hidden_dim,
            moe_shared_expert=moe_shared_expert,
            moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
            moe_loss_weight=moe_loss_weight,
            residual=residual,
            config=config,
        ).to(device)
        model.detach_ymean_in_sum = bool(cfg.get("detach_ymean", getattr(args, "detach_ymean", True)))
    else:
        model = GxE_Transformer(
            g_enc=g_enc,
            e_enc=e_enc,
            ld_enc=ld_enc,
            gxe_enc=gxe_enc,
            moe=cfg.get("wg", cfg.get("moe", args.wg)),
            g_encoder_type=g_encoder_type,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
            moe_expert_hidden_dim=moe_expert_hidden_dim,
            moe_shared_expert=moe_shared_expert,
            moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
            moe_loss_weight=moe_loss_weight,
            config=config,
        ).to(device)

    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, cfg


def _evaluate_checkpoint_on_test(
    checkpoint_path: str,
    args,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, list[tuple[str, float, int]]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    model, cfg = _build_model_from_checkpoint_payload(payload, args, device)

    env_scaler = _rebuild_env_scaler(payload.get("env_scaler"))
    y_scalers = _rebuild_y_scalers(payload.get("y_scalers"))
    marker_stats = _rebuild_marker_stats(payload.get("marker_stats"))
    if env_scaler is None:
        raise ValueError(f"Checkpoint missing env_scaler payload: {checkpoint_path}")

    g_input_type = str(cfg.get("g_input_type", getattr(args, "g_input_type", "tokens"))).lower()
    residual = bool(cfg.get("residual", getattr(args, "residual", False)))
    scale_targets = bool(cfg.get("scale_targets", getattr(args, "scale_targets", True)))

    test_ds = GxE_Dataset(
        split="test",
        data_path="data/maize_data_2014-2023_vs_2024_v2/",
        residual=residual,
        scaler=env_scaler,
        y_scalers=y_scalers,
        scale_targets=scale_targets,
        g_input_type=g_input_type,
        marker_stats=marker_stats,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    all_env = []
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = _move_to_device(xb, device)
            out = model(xb)
            if isinstance(out, dict):
                out = out["total"]
            pred = out.squeeze(-1).detach().cpu().numpy().astype(float)
            if y_scalers and ("total" in y_scalers):
                pred = y_scalers["total"].inverse_transform(pred)

            actual = np.asarray(yb["Yield_Mg_ha"], dtype=float)
            env = np.asarray(yb.get("Env", ["UNK"] * len(pred))).astype(str)

            all_pred.append(pred)
            all_true.append(actual)
            all_env.append(env)

    y_pred = np.concatenate(all_pred) if all_pred else np.asarray([], dtype=float)
    y_true = np.concatenate(all_true) if all_true else np.asarray([], dtype=float)
    env_arr = np.concatenate(all_env) if all_env else np.asarray([], dtype=str)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    env_arr = env_arr[valid]

    if y_true.size == 0:
        return {
            "global_pcc": float("nan"),
            "global_mse": float("nan"),
            "env_pcc": float("nan"),
            "env_mse": float("nan"),
            "env_pcc_weighted": float("nan"),
            "n_test": 0,
        }, []

    global_pcc = _safe_pcc_np(y_true, y_pred)
    global_mse = float(np.mean((y_true - y_pred) ** 2))

    uniq_env, env_ids = np.unique(env_arr, return_inverse=True)
    env_pcc_t = macro_env_pearson(
        pred=torch.tensor(y_pred, dtype=torch.float32, device=device),
        target=torch.tensor(y_true, dtype=torch.float32, device=device),
        env_id=torch.tensor(env_ids, dtype=torch.long, device=device),
        min_samples=2,
    )
    env_pcc = float(env_pcc_t.item()) if bool(torch.isfinite(env_pcc_t).item()) else float("nan")

    pcc_rows: list[tuple[str, float, int]] = []
    weighted_num = 0.0
    weighted_den = 0
    env_mse_vals = []
    for i, env_name in enumerate(uniq_env):
        mask = env_ids == i
        n = int(mask.sum())
        if n < 2:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        pcc_i = _safe_pcc_np(yt, yp)
        mse_i = float(np.mean((yt - yp) ** 2))
        if math.isfinite(pcc_i):
            pcc_rows.append((str(env_name), float(pcc_i), n))
            weighted_num += float(pcc_i) * n
            weighted_den += n
        env_mse_vals.append(mse_i)

    env_pcc_weighted = (weighted_num / weighted_den) if weighted_den > 0 else float("nan")
    env_mse = float(np.mean(env_mse_vals)) if env_mse_vals else float("nan")

    return {
        "global_pcc": global_pcc,
        "global_mse": global_mse,
        "env_pcc": env_pcc,
        "env_mse": env_mse,
        "env_pcc_weighted": env_pcc_weighted,
        "n_test": int(y_true.size),
    }, pcc_rows


def _select_eval_records(
    fold_records: list[dict],
    mode: str,
) -> list[tuple[str, dict]]:
    mode = str(mode).strip().lower()
    if mode in {"", "none", "off", "false", "0", "no"}:
        return []

    def _score(r):
        pcc = r["best_val_env_avg_pearson"]
        loss = r["best_val_loss"]
        pcc_score = pcc if math.isfinite(pcc) else -float("inf")
        loss_score = -loss if math.isfinite(loss) else -float("inf")
        return (pcc_score, loss_score)

    latest = max(fold_records, key=lambda r: r["val_year"])
    best = max(fold_records, key=_score)

    if mode == "best_fold":
        candidates = [("best_fold", best)]
    elif mode == "latest_fold":
        candidates = [("latest_fold", latest)]
    elif mode in {"best_and_latest", "best+latest", "both"}:
        candidates = [("best_fold", best), ("latest_fold", latest)]
    elif mode == "all_folds":
        candidates = [(f"fold_{r['val_year']}", r) for r in sorted(fold_records, key=lambda x: x["val_year"])]
    elif mode.startswith("year:"):
        val_year = int(mode.split(":", 1)[1].strip())
        found = [r for r in fold_records if r["val_year"] == val_year]
        candidates = [(f"fold_{val_year}", found[0])] if found else []
    else:
        raise ValueError(
            "ROLLING_TEST_EVAL_MODE must be one of: "
            "none, best_fold, latest_fold, best_and_latest, all_folds, year:<YYYY>"
        )

    selected = []
    seen = set()
    for tag, rec in candidates:
        ckpt = rec.get("best_checkpoint")
        if not ckpt or ckpt in seen:
            continue
        seen.add(ckpt)
        selected.append((tag, rec))
    return selected


### main ###
def main():
    args = parse_args()
    wandb_run_name = make_run_name(args) + "+rolling"

    device, local_rank, rank, world_size = setup_ddp()

    def _get_arg_or_env(attr, env_key, default, cast=None):
        val = getattr(args, attr, None)
        if val is None:
            env_val = os.getenv(env_key)
            if env_val is None or env_val == "":
                return default
            return cast(env_val) if cast is not None else env_val
        return val

    set_seed(args.seed + rank)

    g_input_type = str(_get_arg_or_env("g_input_type", "G_INPUT_TYPE", "tokens", str)).lower()
    env_stratified = _get_arg_or_env("env_stratified", "ENV_STRATIFIED", False, str2bool)
    min_samples_per_env = _get_arg_or_env("min_samples_per_env", "MIN_SAMPLES_PER_ENV", 32, int)

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

    folds = _build_rolling_folds()
    if not folds:
        raise ValueError("No rolling folds configured. Check ROLLING_* environment variables.")

    rolling_test_eval_mode = str(os.getenv("ROLLING_TEST_EVAL_MODE", "best_fold")).strip().lower()
    rolling_test_batch_size = int(os.getenv("ROLLING_TEST_BATCH_SIZE", "32"))
    rolling_test_primary = str(os.getenv("ROLLING_TEST_PRIMARY", "best_fold")).strip().lower()
    rolling_selected_ckpt_file = os.getenv("ROLLING_SELECTED_CHECKPOINT_FILE", "").strip()

    if is_main(rank):
        print(f"[INFO] Rolling folds: {folds}")

    run_ckpt_dir = Path("checkpoints") / wandb_run_name
    if is_main(rank):
        if run_ckpt_dir.exists():
            shutil.rmtree(run_ckpt_dir)
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    loss_names = build_loss(args.loss, args.loss_weights).names

    run = None
    if is_main(rank):
        run = wandb.init(
            project="gxe-transformer-rolling",
            entity=os.getenv("WANDB_ENTITY"),
            name=wandb_run_name,
        )

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

        # Match non-rolling tracking so overlays are easy.
        run.define_metric("iter_num")
        run.define_metric("train_loss", step_metric="iter_num")
        run.define_metric("learning_rate", step_metric="iter_num")

        run.define_metric("epoch")
        run.define_metric("train_loss_epoch", step_metric="epoch")
        run.define_metric("val_loss", step_metric="epoch")
        run.define_metric("train_loss_epoch/env_avg_pearson", step_metric="epoch")
        run.define_metric("val_loss/env_avg_pearson", step_metric="epoch")
        # aliases to match residual script naming
        run.define_metric("train/env_avg_pearson", step_metric="epoch")
        run.define_metric("val/env_avg_pearson", step_metric="epoch")

        for name in loss_names:
            run.define_metric(f"train_loss/{name}", step_metric="iter_num")
            run.define_metric(f"train_loss_epoch/{name}", step_metric="epoch")
            run.define_metric(f"val_loss/{name}", step_metric="epoch")
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
        if args.residual:
            run.define_metric("aux_ymean_loss", step_metric="iter_num")
            run.define_metric("aux_resid_loss", step_metric="iter_num")
            run.define_metric("aux_ymean_mse_epoch", step_metric="epoch")
            run.define_metric("aux_resid_mse_epoch", step_metric="epoch")
        if moe_encoder_enabled:
            run.define_metric("train_loss/moe_lb", step_metric="iter_num")
            run.define_metric("train_loss_epoch/moe_lb", step_metric="epoch")
            run.define_metric("val_loss/moe_lb", step_metric="epoch")

        # Keep fold metadata separate to avoid noisy primary charts.
        run.define_metric("rolling/fold_idx", step_metric="epoch")
        run.define_metric("rolling/epoch_in_fold", step_metric="epoch")
        run.define_metric("rolling/val_year", step_metric="epoch")
        run.define_metric("rolling/iter_in_fold", step_metric="iter_num")

        wandb.config.update({
            "rolling_folds": folds,
            "loss": args.loss,
            "loss_weights": args.loss_weights,
            "selection_metric": "val_loss/env_avg_pearson",
            "residual": args.residual,
            "full_transformer": args.full_transformer,
            "g_encoder_type": g_encoder_type,
            "full_tf_mlp_type": full_tf_mlp_type,
            "moe_num_experts": moe_num_experts,
            "moe_top_k": moe_top_k,
            "moe_expert_hidden_dim": moe_expert_hidden_dim,
            "moe_shared_expert": moe_shared_expert,
            "moe_shared_expert_hidden_dim": moe_shared_expert_hidden_dim,
            "moe_loss_weight": moe_loss_weight,
            "g_input_type": g_input_type,
            "contrastive_mode": contrastive_mode,
            "env_stratified": env_stratified,
            "min_samples_per_env": min_samples_per_env,
            "batch_size": args.batch_size,
            "gbs": args.gbs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_epochs": args.num_epochs,
            "early_stop": args.early_stop,
            "g_layers": args.g_layers,
            "ld_layers": args.ld_layers,
            "mlp_layers": args.mlp_layers,
            "gxe_layers": args.gxe_layers,
            "heads": args.heads,
            "emb_size": args.emb_size,
            "dropout": args.dropout,
            "scale_targets": args.scale_targets,
            "rolling_test_eval_mode": rolling_test_eval_mode,
            "rolling_test_batch_size": rolling_test_batch_size,
            "rolling_test_primary": rolling_test_primary,
        }, allow_val_change=True)

    fold_records = []
    iter_num_global = 0
    epoch_global = 0

    for fold_idx, (train_year_max, val_year) in enumerate(folds, start=1):
        if is_main(rank):
            print(f"\n=== Fold {fold_idx}/{len(folds)}: train <= {train_year_max}, val = {val_year} ===")

        train_ds = GxE_Dataset(
            split="train",
            data_path='data/maize_data_2014-2023_vs_2024_v2/',
            residual=args.residual,
            scaler=None,
            train_year_max=train_year_max,
            y_scalers=None,
            scale_targets=args.scale_targets,
            g_input_type=g_input_type,
            marker_stats=None,
        )
        env_scaler = train_ds.scaler
        y_scalers = train_ds.label_scalers
        marker_stats = train_ds.marker_stats

        val_ds = GxE_Dataset(
            split="val",
            data_path='data/maize_data_2014-2023_vs_2024_v2/',
            residual=args.residual,
            scaler=env_scaler,
            val_year=val_year,
            y_scalers=y_scalers,
            scale_targets=args.scale_targets,
            g_input_type=g_input_type,
            marker_stats=marker_stats,
        )

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

        use_batch_sampler = False
        if env_stratified and "envpcc" in args.loss.lower():
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

        if use_batch_sampler:
            train_loader = DataLoader(
                train_ds,
                batch_sampler=train_sampler,
                pin_memory=True,
                num_workers=0,
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

        config = Config(
            block_size=train_ds.block_size,
            g_input_type=g_input_type,
            n_head=args.heads,
            n_g_layer=args.g_layers,
            n_ld_layer=args.ld_layers,
            n_mlp_layer=args.mlp_layers,
            n_gxe_layer=args.gxe_layers,
            n_embd=args.emb_size,
            dropout=args.dropout,
            n_env_fts=train_ds.n_env_fts,
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
            model = GxE_ResidualTransformer(
                g_enc=args.g_enc,
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
                config=config,
            ).to(device)
            model.detach_ymean_in_sum = args.detach_ymean
        else:
            model = GxE_Transformer(
                g_enc=args.g_enc,
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
                config=config,
            ).to(device)

        if is_main(rank):
            model.print_trainable_parameters()

        find_unused = (
            bool(args.wg)
            or moe_encoder_enabled
            or bool(args.residual)
            or contrastive_mode in {"g", "e", "g+e"}
        )
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_function = build_loss(args.loss, args.loss_weights)

        batches_per_epoch = len(train_loader)
        effective_epochs = min(args.early_stop * 2, args.num_epochs)
        total_iters = max(1, effective_epochs * max(1, batches_per_epoch))
        warmup_iters = max(1, batches_per_epoch * 5)
        lr_decay_iters = total_iters
        max_lr, min_lr = args.lr, (0.1 * args.lr)
        max_epochs = args.num_epochs

        best_val_loss = float("inf")
        best_val_env_pcc = -float("inf")
        best_ckpt_path = None
        best_epoch = -1
        last_improved = 0
        iter_in_fold = 0

        for epoch_num in range(max_epochs):
            train_sampler.set_epoch(epoch_num)
            model.train()

            pbar = tqdm(total=batches_per_epoch, desc=f"Rank {rank} Fold {val_year}", disable=not is_main(rank))
            for xb, yb in train_loader:
                xb = {k: v.to(device, non_blocking=True) for k, v in xb.items()}
                yb = _move_to_device(yb, device)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    if use_g_contrastive and use_e_contrastive:
                        logits, g_embeddings, e_embeddings = model(
                            xb,
                            return_g_embeddings=True,
                            return_e_embeddings=True,
                        )
                    elif use_g_contrastive:
                        logits, g_embeddings = model(xb, return_g_embeddings=True)
                        e_embeddings = None
                    elif use_e_contrastive:
                        logits, e_embeddings = model(xb, return_e_embeddings=True)
                        g_embeddings = None
                    else:
                        logits = model(xb)
                        g_embeddings = None
                        e_embeddings = None

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

                    contrastive_warmup_epochs = 50
                    if use_g_contrastive or use_e_contrastive:
                        if epoch_num >= contrastive_warmup_epochs:
                            warmup_factor = min(1.0, (epoch_num - contrastive_warmup_epochs) / 50.0)
                            contrastive_total = 0.0

                            if use_g_contrastive and g_embeddings is not None and g_contrastive_loss_fn is not None:
                                g_contr = g_contrastive_loss_fn(g_embeddings, g_data=xb["g_data"])
                                g_weight_eff = contrastive_weight * warmup_factor
                                loss = loss + g_weight_eff * g_contr
                                loss_parts["contrastive_g"] = float(g_contr.detach().item())
                                loss_parts["contrastive_weight_eff_g"] = g_weight_eff
                                contrastive_total += float(g_contr.detach().item())

                            if use_e_contrastive and e_embeddings is not None and e_contrastive_loss_fn is not None:
                                e_contr = e_contrastive_loss_fn(e_embeddings, e_data=xb["e_data"])
                                e_weight_eff = env_contrastive_weight * warmup_factor
                                loss = loss + e_weight_eff * e_contr
                                loss_parts["contrastive_e"] = float(e_contr.detach().item())
                                loss_parts["contrastive_weight_eff_e"] = e_weight_eff
                                contrastive_total += float(e_contr.detach().item())

                            loss_parts["contrastive"] = contrastive_total
                        else:
                            if use_g_contrastive:
                                loss_parts["contrastive_g"] = 0.0
                                loss_parts["contrastive_weight_eff_g"] = 0.0
                            if use_e_contrastive:
                                loss_parts["contrastive_e"] = 0.0
                                loss_parts["contrastive_weight_eff_e"] = 0.0
                            loss_parts["contrastive"] = 0.0

                    moe_aux_loss = getattr(model.module, "moe_aux_loss", None)
                    if moe_aux_loss is not None:
                        loss = loss + moe_aux_loss
                        loss_parts["moe_lb"] = float(moe_aux_loss.detach().item())

                if torch.isnan(loss):
                    raise RuntimeError("Loss is NaN, stopping training.")

                lr = get_lr(iter_in_fold, warmup_iters, lr_decay_iters, max_lr, min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if is_main(rank):
                    payload = {
                        "iter_num": iter_num_global,
                        "train_loss": float(loss.item()),
                        "learning_rate": float(lr),
                        "rolling/iter_in_fold": int(iter_in_fold),
                    }
                    for k, v in loss_parts.items():
                        payload[f"train_loss/{k}"] = float(v)
                    if loss_aux_ymean is not None:
                        payload["aux_ymean_loss"] = float(loss_aux_ymean.detach().item())
                    if loss_aux_resid is not None:
                        payload["aux_resid_loss"] = float(loss_aux_resid.detach().item())
                    wandb.log(payload)

                iter_in_fold += 1
                iter_num_global += 1
                pbar.update(1)
            pbar.close()

            with torch.no_grad():
                model.eval()

                def _gather_predictions(loader, dataset_len):
                    all_preds, all_targets, all_env_ids = [], [], []
                    all_ymean_preds, all_ymean_targets = [], []
                    all_resid_preds, all_resid_targets = [], []

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

                    local_preds = torch.cat(all_preds) if all_preds else torch.empty(0, device=device)
                    local_targets = torch.cat(all_targets) if all_targets else torch.empty(0, device=device)
                    local_env_ids = torch.cat(all_env_ids) if all_env_ids else torch.empty(0, dtype=torch.long, device=device)
                    local_ymean_p = torch.cat(all_ymean_preds) if all_ymean_preds else None
                    local_ymean_t = torch.cat(all_ymean_targets) if all_ymean_targets else None
                    local_resid_p = torch.cat(all_resid_preds) if all_resid_preds else None
                    local_resid_t = torch.cat(all_resid_targets) if all_resid_targets else None

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

                    full_preds = _all_gather_flat(local_preds)[:dataset_len]
                    full_targets = _all_gather_flat(local_targets)[:dataset_len]
                    full_env_ids = _all_gather_flat(local_env_ids.float()).long()[:dataset_len]

                    full_ymean_p = _all_gather_flat(local_ymean_p)[:dataset_len] if local_ymean_p is not None else None
                    full_ymean_t = _all_gather_flat(local_ymean_t)[:dataset_len] if local_ymean_t is not None else None
                    full_resid_p = _all_gather_flat(local_resid_p)[:dataset_len] if local_resid_p is not None else None
                    full_resid_t = _all_gather_flat(local_resid_t)[:dataset_len] if local_resid_t is not None else None

                    return (
                        full_preds,
                        full_targets,
                        full_env_ids,
                        full_ymean_p,
                        full_ymean_t,
                        full_resid_p,
                        full_resid_t,
                    )

                def eval_loader(loader, dataset_len):
                    (
                        full_preds,
                        full_targets,
                        full_env_ids,
                        full_ymean_p,
                        full_ymean_t,
                        full_resid_p,
                        full_resid_t,
                    ) = _gather_predictions(loader, dataset_len)

                    ltot, lparts = loss_function(full_preds, full_targets, env_id=full_env_ids)

                    mean_aux_ymean = 0.0
                    mean_aux_resid = 0.0
                    if full_ymean_p is not None and full_ymean_t is not None:
                        mean_aux_ymean = F.mse_loss(full_ymean_p, full_ymean_t).item()
                    if full_resid_p is not None and full_resid_t is not None:
                        mean_aux_resid = F.mse_loss(full_resid_p, full_resid_t).item()

                    env_pcc = macro_env_pearson(full_preds, full_targets, full_env_ids, min_samples=2)
                    mean_parts = {k: float(v) for k, v in lparts.items()}
                    return ltot, mean_parts, mean_aux_ymean, mean_aux_resid, float(env_pcc.item())

                train_total, train_parts, aux_train_ymean, aux_train_resid, train_env_pcc = eval_loader(train_loader, len(train_ds))
                val_total, val_parts, aux_val_ymean, aux_val_resid, val_env_pcc = eval_loader(val_loader, len(val_ds))

            if is_main(rank):
                payload = {
                    "epoch": epoch_global,
                    "train_loss_epoch": float(train_total.item()),
                    "val_loss": float(val_total.item()),
                    "train_loss_epoch/env_avg_pearson": float(train_env_pcc),
                    "val_loss/env_avg_pearson": float(val_env_pcc),
                    "train/env_avg_pearson": float(train_env_pcc),
                    "val/env_avg_pearson": float(val_env_pcc),
                    "rolling/fold_idx": int(fold_idx),
                    "rolling/epoch_in_fold": int(epoch_num),
                    "rolling/val_year": int(val_year),
                }
                for k, v in train_parts.items():
                    payload[f"train_loss_epoch/{k}"] = float(v)
                for k, v in val_parts.items():
                    payload[f"val_loss/{k}"] = float(v)
                if args.residual:
                    payload["aux_ymean_mse_epoch"] = float(aux_val_ymean)
                    payload["aux_resid_mse_epoch"] = float(aux_val_resid)
                wandb.log(payload)

                val_loss_value = float(val_total.item())
                improved = False
                if math.isfinite(val_env_pcc):
                    if (val_env_pcc > best_val_env_pcc + 1e-8) or (
                        abs(val_env_pcc - best_val_env_pcc) <= 1e-8 and val_loss_value < best_val_loss
                    ):
                        improved = True
                elif not math.isfinite(best_val_env_pcc) and (val_loss_value < best_val_loss):
                    improved = True

                if improved:
                    best_val_loss = val_loss_value
                    best_val_env_pcc = val_env_pcc
                    best_epoch = epoch_num
                    last_improved = 0

                    env_scaler_payload = {
                        "mean": env_scaler.mean_.tolist(),
                        "scale": env_scaler.scale_.tolist(),
                        "var": env_scaler.var_.tolist(),
                        "n_features_in": int(train_ds.scaler.n_features_in_),
                        "feature_names_in": list(train_ds.e_cols),
                    }

                    label_scalers_payload = None
                    if hasattr(train_ds, "label_scalers") and train_ds.label_scalers:
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
                        "model": model.module.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch_num,
                        "fold": fold_idx,
                        "train_year_max": train_year_max,
                        "val_year": val_year,
                        "val_loss": val_loss_value,
                        "val_env_avg_pearson": val_env_pcc,
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
                            "loss": args.loss,
                            "loss_weights": args.loss_weights,
                            "residual": args.residual,
                            "lambda_ymean": args.lambda_ymean,
                            "lambda_resid": args.lambda_resid,
                            "detach_ymean": args.detach_ymean,
                            "scale_targets": args.scale_targets,
                            "contrastive_mode": contrastive_mode,
                            "dropout": args.dropout,
                        },
                        "env_scaler": env_scaler_payload,
                        "y_scalers": label_scalers_payload,
                        "marker_stats": marker_stats_payload,
                        "run": {
                            "id": run.id if run is not None else None,
                            "name": wandb_run_name,
                        },
                    }
                    ckpt_path = run_ckpt_dir / f"fold_{val_year}" / f"checkpoint_{epoch_num:04d}.pt"
                    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(ckpt, ckpt_path)
                    best_ckpt_path = str(ckpt_path)

                    print(
                        f"*** fold {val_year} env_avg_pearson improved: "
                        f"{best_val_env_pcc:.5f} (val_loss={best_val_loss:.4e}) ***"
                    )
                else:
                    last_improved += 1
                    print(f"Fold {val_year} has not improved in {last_improved} epochs")

            epoch_global += 1
            stop_flag = torch.tensor([0], device=device)
            if is_main(rank) and last_improved > args.early_stop:
                print(f"*** fold {val_year}: no improvement for {args.early_stop} epochs, stopping ***")
                stop_flag[0] = 1
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break

            if is_main(rank):
                print(
                    f"[Fold {val_year} Epoch {epoch_num}] train_loss={train_total.item():.4e} | "
                    f"val_loss={val_total.item():.4e} | val_env_pcc={val_env_pcc:.5f} | "
                    f"best_env_pcc={best_val_env_pcc:.5f}"
                )

        if is_main(rank):
            fold_record = {
                "fold_idx": fold_idx,
                "train_year_max": train_year_max,
                "val_year": val_year,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
                "best_val_env_avg_pearson": float(best_val_env_pcc),
                "best_checkpoint": best_ckpt_path,
            }
            fold_records.append(fold_record)
            wandb.log({
                "rolling/fold/final/val_year": int(val_year),
                "rolling/fold/final/best_epoch": int(best_epoch),
                "rolling/fold/final/best_val_loss": float(best_val_loss),
                "rolling/fold/final/best_val_env_avg_pearson": float(best_val_env_pcc),
            })

        dist.barrier()
        del model
        torch.cuda.empty_cache()

    if is_main(rank) and fold_records:
        mean_val_loss = float(np.mean([r["best_val_loss"] for r in fold_records]))
        std_val_loss = float(np.std([r["best_val_loss"] for r in fold_records]))
        mean_val_env_pcc = float(np.mean([r["best_val_env_avg_pearson"] for r in fold_records]))
        std_val_env_pcc = float(np.std([r["best_val_env_avg_pearson"] for r in fold_records]))

        def _score(r):
            pcc = r["best_val_env_avg_pearson"]
            loss = r["best_val_loss"]
            pcc_score = pcc if math.isfinite(pcc) else -float("inf")
            loss_score = -loss if math.isfinite(loss) else -float("inf")
            return (pcc_score, loss_score)

        best_fold = max(fold_records, key=_score)
        latest_fold = max(fold_records, key=lambda r: r["val_year"])

        table = wandb.Table(columns=[
            "fold_idx",
            "train_year_max",
            "val_year",
            "best_epoch",
            "best_val_loss",
            "best_val_env_avg_pearson",
            "best_checkpoint",
        ])
        for r in fold_records:
            table.add_data(
                r["fold_idx"],
                r["train_year_max"],
                r["val_year"],
                r["best_epoch"],
                r["best_val_loss"],
                r["best_val_env_avg_pearson"],
                r["best_checkpoint"],
            )

        wandb.log({
            "cv/folds_table": table,
            "cv/mean_val_loss": mean_val_loss,
            "cv/std_val_loss": std_val_loss,
            "cv/mean_val_env_avg_pearson": mean_val_env_pcc,
            "cv/std_val_env_avg_pearson": std_val_env_pcc,
            "cv/best_fold_val_year": best_fold["val_year"],
            "cv/best_fold_env_avg_pearson": best_fold["best_val_env_avg_pearson"],
            "cv/best_fold_checkpoint": best_fold["best_checkpoint"],
            "cv/latest_fold_val_year": latest_fold["val_year"],
            "cv/latest_fold_checkpoint": latest_fold["best_checkpoint"],
        })

        run.summary["cv/mean_val_loss"] = mean_val_loss
        run.summary["cv/std_val_loss"] = std_val_loss
        run.summary["cv/mean_val_env_avg_pearson"] = mean_val_env_pcc
        run.summary["cv/std_val_env_avg_pearson"] = std_val_env_pcc
        run.summary["cv/best_fold_val_year"] = best_fold["val_year"]
        run.summary["cv/best_fold_env_avg_pearson"] = best_fold["best_val_env_avg_pearson"]
        run.summary["cv/best_fold_checkpoint"] = best_fold["best_checkpoint"]
        run.summary["cv/latest_fold_val_year"] = latest_fold["val_year"]
        run.summary["cv/latest_fold_checkpoint"] = latest_fold["best_checkpoint"]

        run.summary["rolling/test_eval_mode"] = rolling_test_eval_mode
        run.summary["rolling/test_batch_size"] = int(rolling_test_batch_size)
        run.summary["rolling/test_primary"] = rolling_test_primary

        eval_records = _select_eval_records(fold_records, rolling_test_eval_mode)

        def _finite_or_none(x):
            x = float(x)
            return x if math.isfinite(x) else None

        if eval_records:
            eval_metrics = {}
            eval_record_map = {}
            for tag, rec in eval_records:
                ckpt_path = rec["best_checkpoint"]
                if not ckpt_path:
                    print(f"[WARN] No checkpoint path for selection tag '{tag}', skipping test eval.")
                    continue
                print(f"[TEST] Evaluating {tag} checkpoint: {ckpt_path}")
                try:
                    metrics, pcc_rows = _evaluate_checkpoint_on_test(
                        checkpoint_path=ckpt_path,
                        args=args,
                        device=device,
                        batch_size=rolling_test_batch_size,
                    )
                except Exception as e:
                    print(f"[WARN] Test evaluation failed for {tag} ({ckpt_path}): {e}")
                    continue

                eval_metrics[tag] = metrics
                eval_record_map[tag] = rec

                tag_payload = {
                    f"rolling/test/{tag}/pearson": _finite_or_none(metrics["global_pcc"]),
                    f"rolling/test/{tag}/mse": _finite_or_none(metrics["global_mse"]),
                    f"rolling/test/{tag}/env_avg_pearson": _finite_or_none(metrics["env_pcc"]),
                    f"rolling/test/{tag}/env_avg_mse": _finite_or_none(metrics["env_mse"]),
                    f"rolling/test/{tag}/env_avg_pearson_weighted": _finite_or_none(metrics["env_pcc_weighted"]),
                    f"rolling/test/{tag}/n_test": int(metrics["n_test"]),
                }
                wandb.log({k: v for k, v in tag_payload.items() if v is not None})
                for k, v in tag_payload.items():
                    if v is not None:
                        run.summary[k] = v
                run.summary[f"rolling/test/{tag}/checkpoint"] = ckpt_path
                run.summary[f"rolling/test/{tag}/val_year"] = int(rec["val_year"])
                run.summary[f"rolling/test/{tag}/best_val_env_avg_pearson"] = float(rec["best_val_env_avg_pearson"])

                if pcc_rows:
                    pcc_table = wandb.Table(columns=["Env", "PCC", "Count"])
                    for env_name, env_pcc_i, env_count in pcc_rows:
                        pcc_table.add_data(env_name, float(env_pcc_i), int(env_count))
                    wandb.log({f"rolling/test/{tag}/pcc_by_env": pcc_table})

                print(
                    f"[TEST] {tag}: "
                    f"env_avg_pearson={metrics['env_pcc']:.5f}, "
                    f"weighted_env_avg_pearson={metrics['env_pcc_weighted']:.5f}, "
                    f"mse={metrics['global_mse']:.5f}, n={metrics['n_test']}"
                )

            primary_tag = None
            if rolling_test_primary in eval_metrics:
                primary_tag = rolling_test_primary
            elif "best_fold" in eval_metrics:
                primary_tag = "best_fold"
            elif len(eval_metrics):
                primary_tag = next(iter(eval_metrics.keys()))

            if primary_tag is not None:
                primary_metrics = eval_metrics[primary_tag]
                primary_rec = eval_record_map[primary_tag]
                primary_payload = {
                    "test/pearson": _finite_or_none(primary_metrics["global_pcc"]),
                    "test/mse": _finite_or_none(primary_metrics["global_mse"]),
                    "test/env_avg_pearson": _finite_or_none(primary_metrics["env_pcc"]),
                    "test/env_avg_mse": _finite_or_none(primary_metrics["env_mse"]),
                    "test/env_avg_pearson_weighted": _finite_or_none(primary_metrics["env_pcc_weighted"]),
                }
                wandb.log({k: v for k, v in primary_payload.items() if v is not None})
                for k, v in primary_payload.items():
                    if v is not None:
                        run.summary[k] = v
                run.summary["test/source"] = f"rolling:{primary_tag}"
                run.summary["test/source_checkpoint"] = primary_rec["best_checkpoint"]
                run.summary["test/source_val_year"] = int(primary_rec["val_year"])
                run.summary["test/source_best_val_env_avg_pearson"] = float(primary_rec["best_val_env_avg_pearson"])

                if rolling_selected_ckpt_file:
                    try:
                        Path(rolling_selected_ckpt_file).parent.mkdir(parents=True, exist_ok=True)
                        with open(rolling_selected_ckpt_file, "w") as f:
                            f.write(primary_rec["best_checkpoint"])
                        print(f"[INFO] Selected rolling checkpoint written to {rolling_selected_ckpt_file}")
                    except Exception as e:
                        print(f"[WARN] Failed writing selected checkpoint file '{rolling_selected_ckpt_file}': {e}")

    dist.barrier()
    if is_main(rank) and run is not None:
        run.finish()
    cleanup_ddp()


if __name__ == "__main__":
    main()
