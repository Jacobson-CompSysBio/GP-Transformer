# import packages 
import json
import os, sys
import wandb
import argparse
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.dataset import *
from models.config import *
from models.model import *
from utils.loss import macro_env_pearson, macro_env_ccc
from utils.utils import *

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = 'data/maize_data_2014-2023_vs_2024_v2/'

# safer pcc to avoid nans
def _safe_pcc(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return np.nan
    # if our data is constant, return nan
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    r = pearsonr(a, b)[0]
    return r # robust to old/new scipy versions


def _rebuild_env_scaler(payload: dict) -> StandardScaler | None:
    if not payload:
        return None
    scaler = StandardScaler()
    scaler.mean_ = np.array(payload['mean'], dtype=float)
    scaler.scale_ = np.array(payload['scale'], dtype=float)
    scaler.var_ = np.array(payload['var'], dtype=float)
    scaler.n_features_in_ = int(payload['n_features_in'])
    if 'feature_names_in' in payload:
        scaler.feature_names_in_ = np.array(payload['feature_names_in'], dtype=object)
    return scaler

def _rebuild_y_scalers(payload: dict) -> dict | None:
    if not payload:
        return None
    return {k: LabelScaler(v['mean'], v['std']) for k, v in payload.items()}

def _rebuild_marker_stats(payload: dict) -> dict | None:
    if not payload:
        return None
    return {
        "p": np.asarray(payload["p"], dtype=np.float32),
        "scale": np.asarray(payload["scale"], dtype=np.float32),
        "valid": np.asarray(payload["valid"], dtype=np.float32),
        "columns": list(payload["columns"]),
    }


def _inverse_transform_prediction(pred: torch.Tensor, y_scalers: dict | None) -> np.ndarray:
    if y_scalers and 'total' in y_scalers:
        return np.asarray(y_scalers['total'].inverse_transform(pred), dtype=float).ravel()
    return pred.detach().cpu().numpy().ravel()


def _resolve_checkpoint_path(args) -> Path:
    checkpoint_path = getattr(args, "checkpoint_path", None)
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist.")
        return ckpt_path

    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if checkpoint_dir is None:
        raise ValueError("Either --checkpoint_path or --checkpoint_dir must be provided.")

    ckpt_root = Path(checkpoint_dir).expanduser().resolve()
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_root} does not exist.")

    checkpoint_tag = getattr(args, "checkpoint_tag", None)
    if checkpoint_tag:
        manifest_path = ckpt_root / "checkpoint_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            alias_info = manifest.get("aliases", {}).get(checkpoint_tag)
            if alias_info:
                candidate = Path(alias_info["path"])
                if not candidate.is_absolute():
                    candidate = ckpt_root / candidate
                if candidate.exists():
                    return candidate.resolve()
        fallback = ckpt_root / f"{checkpoint_tag}.pt"
        if fallback.exists():
            return fallback.resolve()
        raise FileNotFoundError(
            f"Could not resolve checkpoint_tag={checkpoint_tag!r} in {ckpt_root}."
        )

    ckpts = sorted([p for p in ckpt_root.glob("checkpoint_*.pt") if p.is_file()])
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_root}.")
    return ckpts[-1].resolve()


def _resolve_metric_prefix(args) -> str:
    if getattr(args, "metric_prefix", None):
        return str(args.metric_prefix)
    checkpoint_tag = getattr(args, "checkpoint_tag", None)
    if checkpoint_tag and checkpoint_tag != "best_leo":
        return f"test_{checkpoint_tag}"
    return "test"

def load_data(args,
              split: str = "test",
              batch_size: int = 32,
              env_scaler: StandardScaler | None = None,
              y_scalers: dict | None = None,
              marker_stats: dict | None = None,
              g_input_type: str = "tokens",
              env_categorical_mode: str = "drop"):

    ds = GxE_Dataset(
        split=split,
        data_path=DATA_DIR,
        scaler=env_scaler,
        y_scalers=y_scalers,                           # <— pass dict here
        scale_targets=args.scale_targets,
        g_input_type=g_input_type,
        env_categorical_mode=env_categorical_mode,
        marker_stats=marker_stats,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return ds, loader

def evaluate(model,
             test_loader: DataLoader,
             y_scalers: dict | None,
             device: torch.device):

    model.eval()
    rows = []

    with torch.no_grad():
        for xb, yb in tqdm(test_loader):
            # get things on device
            for key, value in xb.items():
                xb[key] = value.to(device, non_blocking=True)

            # forward pass
            out = model(xb)
            if isinstance(out, dict):
                total_out = out.get('total')
                rank_out = out.get('rank', total_out)
                scale_out = out.get('scale')
                shift_out = out.get('shift')
            else:
                total_out = out
                rank_out = out
                scale_out = None
                shift_out = None

            total_pred = _inverse_transform_prediction(total_out, y_scalers)
            rank_pred = _inverse_transform_prediction(rank_out, y_scalers)
            scale_pred = (
                scale_out.detach().cpu().numpy().ravel()
                if scale_out is not None else np.full_like(total_pred, np.nan, dtype=float)
            )
            shift_pred = (
                shift_out.detach().cpu().numpy().ravel()
                if shift_out is not None else np.full_like(total_pred, np.nan, dtype=float)
            )

            has_y = ('Yield_Mg_ha' in yb)
            if has_y:
                actual = np.asarray(yb['Yield_Mg_ha'], dtype=float)
            else:
                actual = np.full_like(total_pred, fill_value=np.nan, dtype=float)

            env = np.asarray(yb['Env']).astype(str) if 'Env' in yb else np.array(['UNK'] * len(total_pred))
            rows.extend(
                zip(
                    env.tolist(),
                    actual.tolist(),
                    total_pred.tolist(),
                    rank_pred.tolist(),
                    scale_pred.tolist(),
                    shift_pred.tolist(),
                )
            )
    
    # convert our multi-dimensional list to a dataframe
    df = pd.DataFrame(rows, columns=['Env', 'Actual', 'Pred', 'RankPred', 'Scale', 'Shift'])

    # filter to finite pairs for metric computation
    y_true = df['Actual'].to_numpy(dtype=float)
    y_pred = df['Pred'].to_numpy(dtype=float)
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    df_valid = df.loc[valid_mask].copy()

    if len(df_valid):
        # global results
        y_true_valid = df_valid['Actual'].to_numpy(dtype=float)
        y_pred_valid = df_valid['Pred'].to_numpy(dtype=float)
        y_rank_valid = df_valid['RankPred'].to_numpy(dtype=float)
        global_pcc = _safe_pcc(y_true_valid, y_pred_valid)
        global_mse = float(mean_squared_error(y_true_valid, y_pred_valid))

        # env-avg Pearson via shared helper (same as training selection metric)
        env_ids = pd.Categorical(df_valid['Env'].astype(str)).codes
        rank_env_pcc_t = macro_env_pearson(
            pred=torch.tensor(y_rank_valid, dtype=torch.float32, device=device),
            target=torch.tensor(y_true_valid, dtype=torch.float32, device=device),
            env_id=torch.tensor(env_ids, dtype=torch.long, device=device),
            min_samples=2,
        )
        macro_env_pcc_t = macro_env_pearson(
            pred=torch.tensor(y_pred_valid, dtype=torch.float32, device=device),
            target=torch.tensor(y_true_valid, dtype=torch.float32, device=device),
            env_id=torch.tensor(env_ids, dtype=torch.long, device=device),
            min_samples=2,
        )
        rank_env_pcc = (
            float(rank_env_pcc_t.item())
            if bool(torch.isfinite(rank_env_pcc_t).item())
            else np.nan
        )
        macro_env_pcc = (
            float(macro_env_pcc_t.item())
            if bool(torch.isfinite(macro_env_pcc_t).item())
            else np.nan
        )
        macro_env_ccc_t = macro_env_ccc(
            pred=torch.tensor(y_pred_valid, dtype=torch.float32, device=device),
            target=torch.tensor(y_true_valid, dtype=torch.float32, device=device),
            env_id=torch.tensor(env_ids, dtype=torch.long, device=device),
            min_samples=2,
        )
        macro_env_ccc_v = (
            float(macro_env_ccc_t.item())
            if bool(torch.isfinite(macro_env_ccc_t).item())
            else np.nan
        )

        # per-env tables for reporting
        grp = df_valid.groupby('Env', sort=False)[['Actual', 'Pred']]
        pcc_by_env = grp.apply(
            lambda g: _safe_pcc(g['Actual'].to_numpy(), g['Pred'].to_numpy()),
            include_groups=False
        ).dropna()
        counts = grp.size().loc[pcc_by_env.index]
        weighted_env_pcc = (
            float(np.nansum(pcc_by_env.values * counts.values) / counts.values.sum())
            if len(pcc_by_env)
            else np.nan
        )

        mse_by_env = grp.apply(
            lambda g: float(mean_squared_error(g['Actual'].to_numpy(), g['Pred'].to_numpy())),
            include_groups=False
        ).dropna()
        macro_env_mse = float(mse_by_env.mean()) if len(mse_by_env) else np.nan

        scale_std_by_env = (df_valid.groupby('Env', sort=False)['Scale']
                            .std(ddof=0)
                            .dropna())
        shift_std_by_env = (df_valid.groupby('Env', sort=False)['Shift']
                            .std(ddof=0)
                            .dropna())
        max_env_scale_std = float(scale_std_by_env.max()) if len(scale_std_by_env) else 0.0
        max_env_shift_std = float(shift_std_by_env.max()) if len(shift_std_by_env) else 0.0
    else:
        global_pcc = np.nan
        global_mse = np.nan
        rank_env_pcc = np.nan
        macro_env_pcc = np.nan
        macro_env_ccc_v = np.nan
        weighted_env_pcc = np.nan
        macro_env_mse = np.nan
        pcc_by_env = pd.Series(dtype=float)
        max_env_scale_std = 0.0
        max_env_shift_std = 0.0

    if 'Scale' in df and df['Scale'].notna().any() and np.isfinite(rank_env_pcc) and np.isfinite(macro_env_pcc):
        if abs(rank_env_pcc - macro_env_pcc) > 1e-5:
            raise RuntimeError(
                "rank_env_pcc and total_env_pcc diverged under env_affine calibration: "
                f"{rank_env_pcc:.8f} vs {macro_env_pcc:.8f}"
            )
        if max(max_env_scale_std, max_env_shift_std) > 1e-5:
            raise RuntimeError(
                "env_affine calibration produced per-environment-varying scale/shift: "
                f"max_scale_std={max_env_scale_std:.8e}, max_shift_std={max_env_shift_std:.8e}"
            )

    results = {
        'global_pcc': float(global_pcc) if not np.isnan(global_pcc) else None,
        'global_mse': global_mse,
        'env_pcc': macro_env_pcc,
        'rank_env_pcc': rank_env_pcc,
        'total_env_pcc': macro_env_pcc,
        'env_ccc': macro_env_ccc_v,
        'env_mse': macro_env_mse,
        'env_pcc_weighted': weighted_env_pcc,
        'pcc_by_env': pcc_by_env,
        'max_env_scale_std': max_env_scale_std,
        'max_env_shift_std': max_env_shift_std,
    }

    return results, df

def plot_results(model_type: str,
                 actuals: list,
                 preds: list):

    #find line of best fit
    actuals = np.array(actuals, dtype=float).ravel()
    preds = np.array(preds, dtype=float)

    # if preds is 2d, take first col, else, flatten
    if preds.ndim > 1:
        preds = preds.reshape(len(preds), -1)[:, 0]
    else:
        preds = preds.ravel()

    a, b = np.polyfit(actuals, preds, 1)

    #add points to plot
    plt.scatter(actuals, preds)

    #add line of best fit to plot
    plt.plot(actuals, a*actuals+b, color="orange")

    plt.title("Predicted vs. Actual Maize Yield")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    out_path = RESULTS_DIR / f"{model_type}_pred_plot.png"
    plt.savefig(out_path)
    plt.close()
    
    return out_path


def save_results(model_type: str, df: pd.DataFrame, out_dir: Path = RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    # df must have ['Env','Actual','Pred']
    if not set(['Env','Actual','Pred']).issubset(df.columns):
        raise ValueError("save_results expects a dataframe with columns: Env, Actual, Pred")

    def _safe(g):
        return _safe_pcc(g['Actual'].to_numpy(), g['Pred'].to_numpy())

    by_env = (df.groupby('Env', sort=False)
                .apply(lambda g: _safe(g))
                .dropna()
                .rename('pearson')
                .reset_index())
    by_env = by_env.sort_values('pearson', ascending=False).reset_index(drop=True)
    out_path = out_dir / f"{model_type}_location_results.csv"
    by_env.to_csv(out_path, index=False)
    return out_path

def load_model(device: torch.device,
               args):
    best_path = _resolve_checkpoint_path(args)
    print(f"Loading model from {best_path}")
    payload = torch.load(best_path, map_location="cpu")
    state = payload["model"]
    config = payload.get("config", {})

    g_enc = config.get("g_enc", args.g_enc)
    e_enc = config.get("e_enc", args.e_enc)
    ld_enc = config.get("ld_enc", args.ld_enc)
    gxe_enc = config.get("gxe_enc", args.gxe_enc)
    blk = config.get("block_size", None)
    n_env_fts = config.get("n_env_fts", None)
    g_layer = config.get("g_layers", args.g_layers)
    ld_layer = config.get("ld_layers", args.ld_layers)
    mlp_layer = config.get("mlp_layers", args.mlp_layers)
    gxe_layer = config.get("gxe_layers", args.gxe_layers)
    n_head = config.get("n_head", args.heads)
    n_embd = config.get("n_embd", args.emb_size)
    # Support both old "moe" key and new "wg" key for backwards compatibility
    moe = config.get("wg", config.get("moe", args.wg))
    loss = config.get("loss", args.loss)
    loss_weights = config.get("loss_weights", args.loss_weights)
    residual = config.get("residual", args.residual)
    full_transformer = config.get("full_transformer", getattr(args, "full_transformer", False))
    g_encoder_type = config.get("g_encoder_type", getattr(args, "g_encoder_type", "dense"))
    moe_num_experts = config.get("moe_num_experts", getattr(args, "moe_num_experts", 4))
    moe_top_k = config.get("moe_top_k", getattr(args, "moe_top_k", 2))
    moe_expert_hidden_dim = config.get("moe_expert_hidden_dim", getattr(args, "moe_expert_hidden_dim", None))
    moe_shared_expert = config.get("moe_shared_expert", getattr(args, "moe_shared_expert", False))
    moe_shared_expert_hidden_dim = config.get("moe_shared_expert_hidden_dim", getattr(args, "moe_shared_expert_hidden_dim", None))
    moe_loss_weight = config.get("moe_loss_weight", getattr(args, "moe_loss_weight", 0.01))
    g_input_type = config.get("g_input_type", getattr(args, "g_input_type", "tokens"))
    env_categorical_mode = config.get(
        "env_categorical_mode",
        ("onehot" if bool(config.get("env_cat_embeddings", False)) else getattr(args, "env_categorical_mode", "drop")),
    )
    env_categorical_mode = normalize_env_categorical_mode(env_categorical_mode)
    full_tf_mlp_type = config.get("full_tf_mlp_type", getattr(args, "full_tf_mlp_type", None))
    calibration_mode = config.get("calibration_mode", getattr(args, "calibration_mode", "none"))
    if full_tf_mlp_type is None:
        full_tf_mlp_type = g_encoder_type
    if isinstance(full_tf_mlp_type, str):
        full_tf_mlp_type = full_tf_mlp_type.lower()
    else:
        full_tf_mlp_type = "moe" if full_tf_mlp_type else "dense"

    # build scalers
    env_scaler = _rebuild_env_scaler(payload.get("env_scaler", None))
    y_scalers = _rebuild_y_scalers(payload.get("y_scalers", None))
    marker_stats = _rebuild_marker_stats(payload.get("marker_stats", None))

    config = Config(block_size=blk,
                    g_input_type=g_input_type,
                    n_g_layer=g_layer,
                    n_ld_layer=ld_layer,
                    n_mlp_layer=mlp_layer,
                    n_gxe_layer=gxe_layer,
                    n_head=n_head,
                    n_embd=n_embd,
                    n_env_fts=n_env_fts)
    config.calibration_mode = calibration_mode
    # stash MoE settings so downstream components can read from config if needed
    config.g_encoder_type = g_encoder_type
    config.moe_num_experts = moe_num_experts
    config.moe_top_k = moe_top_k
    config.moe_expert_hidden_dim = moe_expert_hidden_dim
    config.moe_shared_expert = moe_shared_expert
    config.moe_shared_expert_hidden_dim = moe_shared_expert_hidden_dim
    config.moe_loss_weight = moe_loss_weight
    config.full_tf_mlp_type = full_tf_mlp_type
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
    elif residual:
        model = GxE_ResidualTransformer(g_enc=g_enc,
                                        e_enc=e_enc,
                                        ld_enc=ld_enc,
                                        gxe_enc=gxe_enc,
                                        moe=moe,
                                        g_encoder_type=g_encoder_type,
                                        moe_num_experts=moe_num_experts,
                                        moe_top_k=moe_top_k,
                                        moe_expert_hidden_dim=moe_expert_hidden_dim,
                                        moe_shared_expert=moe_shared_expert,
                                        moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
                                        moe_loss_weight=moe_loss_weight,
                                        residual=residual,
                                        config=config).to(device)
        model.detach_ymean_in_sum = args.detach_ymean
    else:
        model = GxE_Transformer(g_enc=g_enc,
                                e_enc=e_enc,
                                ld_enc=ld_enc,
                                gxe_enc=gxe_enc,
                                moe=moe,
                                g_encoder_type=g_encoder_type,
                                moe_num_experts=moe_num_experts,
                                moe_top_k=moe_top_k,
                                moe_expert_hidden_dim=moe_expert_hidden_dim,
                                moe_shared_expert=moe_shared_expert,
                                moe_shared_expert_hidden_dim=moe_shared_expert_hidden_dim,
                                moe_loss_weight=moe_loss_weight,
                                config=config).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, y_scalers, env_scaler, marker_stats, g_input_type, env_categorical_mode 

def main():
    args = parse_args()

    model_type = resolve_model_type(args)
    model_base_type, _ = split_runtime_suffix(model_type)
    metric_prefix = _resolve_metric_prefix(args)

    # set up wand tracking
    load_dotenv()
    os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
    os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    # resume the run from exported slurm id
    run_kwargs = dict(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
    )
    run_id = os.getenv("WANDB_RUN_ID")
    if run_id:
        run_kwargs.update(dict(id=run_id, resume="allow"))
        print(f"[INFO] Resuming WandB run with id: {run_id} | Appending eval metrics")
    else:
        print("[WARNING] No WandB run ID found. Starting a new run.")
        run_kwargs.update(dict(name=f"eval_{model_type}"))
    run = wandb.init(**run_kwargs)

    # load model
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    set_seed(args.seed)
    print("Loading model...")
    model, y_scalers, env_scaler, marker_stats, g_input_type, env_categorical_mode = load_model(device, args)    
    
    # load data
    print("Loading data...")
    test_data, test_loader = load_data(args,
                                    env_scaler=env_scaler,
                                    y_scalers=y_scalers,   # <— name matches
                                    marker_stats=marker_stats,
                                    g_input_type=g_input_type,
                                    env_categorical_mode=env_categorical_mode,
                                    split="test",
                                    batch_size=32)

    # evaluate
    print("Evaluating model...")
    results, df = evaluate(model, test_loader, y_scalers, device)
    plot_path = plot_results(model_type, df['Actual'], df['Pred'])
    #save_results(model_type, actuals, preds)

    # print results
    print("Pearson Correlation:", results['global_pcc'])
    print("Mean Squared Error:", results['global_mse'])
    print("Rank Environment-Averaged Pearson Correlation:", results['rank_env_pcc'])
    print("Total Environment-Averaged Pearson Correlation:", results['total_env_pcc'])
    print("Environment-Averaged CCC:", results['env_ccc'])
    print("Environment-Averaged MSE:", results['env_mse'])
    print("Weighted Environment-Averaged Pearson Correlation:", results['env_pcc_weighted'])
    for pcc in results['pcc_by_env'].items():
        print(f"Env PCC: {pcc}")

    # log metrics
    run.summary[f"{metric_prefix}/pearson"] = float(results['global_pcc'])
    run.summary[f"{metric_prefix}/mse"] = float(results['global_mse'])
    run.summary[f"{metric_prefix}/env_avg_pearson"] = float(results['total_env_pcc'])
    run.summary[f"{metric_prefix}/rank_env_pcc"] = float(results['rank_env_pcc'])
    run.summary[f"{metric_prefix}/total_env_pcc"] = float(results['total_env_pcc'])
    run.summary[f"{metric_prefix}/env_avg_ccc"] = float(results['env_ccc'])
    run.summary[f"{metric_prefix}/env_avg_mse"] = float(results['env_mse'])
    run.summary[f"{metric_prefix}/env_avg_pearson_weighted"] = float(results['env_pcc_weighted'])
    run.summary[f"{metric_prefix}/max_env_scale_std"] = float(results['max_env_scale_std'])
    run.summary[f"{metric_prefix}/max_env_shift_std"] = float(results['max_env_shift_std'])
    run.summary[f"{metric_prefix}/checkpoint_tag"] = getattr(args, "checkpoint_tag", None) or "latest_numeric"
    run.summary[f"{metric_prefix}/model_type"] = model_type
    run.summary[f"{metric_prefix}/model_base_type"] = model_base_type
    
    # table for pcc by env
    pcc_series = results['pcc_by_env']
    pcc_df = pcc_series.rename("PCC").reset_index()
    pcc_df["Env"] = pcc_df["Env"].astype(str)
    pcc_df = pcc_df.replace({np.nan: None})

    pcc_table = wandb.Table(dataframe=pcc_df)
    run.log({f"{metric_prefix}/pcc_by_env": pcc_table})

    run.log({
        f"{metric_prefix}/pearson": float(results['global_pcc']),
        f"{metric_prefix}/env_avg_pearson": float(results['total_env_pcc']),
        f"{metric_prefix}/rank_env_pcc": float(results['rank_env_pcc']),
        f"{metric_prefix}/total_env_pcc": float(results['total_env_pcc']),
        f"{metric_prefix}/env_avg_ccc": float(results['env_ccc']),
        f"{metric_prefix}/env_avg_pearson_weighted": float(results['env_pcc_weighted']),
        f"{metric_prefix}/mse": float(results['global_mse']),
        f"{metric_prefix}/env_avg_mse": float(results['env_mse']),
        f"{metric_prefix}/max_env_scale_std": float(results['max_env_scale_std']),
        f"{metric_prefix}/max_env_shift_std": float(results['max_env_shift_std']),
    })

    # plot image
    run.log({f"{metric_prefix}/pred_vs_actual": wandb.Image(str(plot_path))})
    run.finish()

if __name__ == "__main__":
    main()
