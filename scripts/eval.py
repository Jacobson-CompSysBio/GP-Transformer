# import packages 
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
from utils.utils import parse_args, make_run_name, LabelScaler

import random
random.seed(0); np.random.seed(0); torch.manual_seed(0)

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = 'data/maize_data_2014-2023_vs_2024/'
INDEX_MAP = 'data/maize_data_2014-2023_vs_2024/'

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

def load_data(args,
              split: str = "test",
              batch_size: int = 32,
              env_scaler: StandardScaler | None = None,
              scaler: LabelScaler | None = None):
    # get scaler 
    ds = GxE_Dataset(
        split=split,
        data_path=DATA_DIR,
        index_map_path=INDEX_MAP + 'location_2024.csv',
        scaler=env_scaler,
        y_scalers=scaler,
        scale_targets=args.scale_targets

    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return ds, loader 

def evaluate(model,
             test_loader: DataLoader,
             y_scalers: dict | None,
             device: torch.device):

    preds = []
    actuals = []
    model.eval()

    with torch.no_grad():
    # loop through
        for xb, yb in tqdm(test_loader):
            # get things on device
            for key, value in xb.items():
                xb[key] = value.to(device, non_blocking=True)
            out = model(xb)
            if isinstance(out, dict):
                out = out['total'] 
            if y_scalers and 'total' in y_scalers:
                out = y_scalers['total'].inverse_transform(out)
            out = out.detach().tolist() if isinstance(out, torch.Tensor) else out.tolist()
            preds.extend(out)
            actuals.extend(yb['Yield_Mg_ha'])

    return actuals, preds

def plot_results(model_type: str,
                 actuals: list,
                 preds: list):

    #find line of best fit
    actuals = np.array(actuals)
    preds = np.array(preds).squeeze(-1)
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


def save_results(
        model_type: str,
        actuals: list,
        preds: list):

    # get locations
    locations = pd.read_csv('data/maize_data_2014-2023_vs_2024/location_2024.csv')
    location_names = list(np.unique(locations['Env']))
    locations['actual'] = actuals
    locations['pred'] = preds

    location_results_df = pd.DataFrame({'location':[],
                                        'pearson':[]})
    for location in location_names:
        subset = locations[locations['Env'] == location]
        pcc = pearsonr(subset['actual'], subset['pred'])[0]
        new_result = pd.DataFrame({'location': [location], 'pearson': [pcc]})
        location_results_df = pd.concat([location_results_df, new_result])
    location_results_df = location_results_df.reset_index(drop=True)
    location_results_df.to_csv(f'data/results/{model_type}_location_results.csv')

def load_model(device: torch.device,
               args):

    ckpt_root = Path(args.checkpoint_dir)
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_root} does not exist.")

    ckpts = sorted([p for p in ckpt_root.glob("checkpoint_*.pt") if p.is_file()])
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_root}.")
    
    best_path = ckpts[-1]
    print(f"Loading model from {best_path}")
    payload = torch.load(best_path, map_location="cpu")
    state = payload["model"]
    config = payload.get("config", {})

    g_enc = config.get("g_enc", args.g_enc)
    e_enc = config.get("e_enc", args.e_enc)
    ld_enc = config.get("ld_enc", args.ld_enc)
    gxe_enc = config.get("gxe_enc", args.gxe_enc)
    blk = config.get("block_size", 2240)
    g_layer = config.get("g_layers", args.g_layers)
    ld_layer = config.get("ld_layers", args.ld_layers)
    mlp_layer = config.get("mlp_layers", args.mlp_layers)
    gxe_layer = config.get("gxe_layers", args.gxe_layers)
    n_head = config.get("n_head", args.heads)
    n_embd = config.get("n_embd", args.emb_size)
    moe = config.get("moe", args.moe)
    loss = config.get("loss", args.loss)
    alpha = config.get("alpha", args.alpha)
    residual = config.get("residual", args.residual)

    # build scalers
    env_scaler = _rebuild_env_scaler(payload.get("env_scaler", None))
    y_scalers = _rebuild_y_scalers(payload.get("y_scalers", None))

    config = Config(block_size=blk,
                    n_g_layer=g_layer,
                    n_ld_layer=ld_layer,
                    n_mlp_layer=mlp_layer,
                    n_gxe_layer=gxe_layer,
                    n_head=n_head,
                    n_embd=n_embd)
    if args.residual:
        model = GxE_ResidualTransformer(g_enc=g_enc,
                                        e_enc=e_enc,
                                        ld_enc=ld_enc,
                                        gxe_enc=gxe_enc,
                                        moe=moe,
                                        residual=residual,
                                        config=config).to(device)
        model.detach_ymean_in_sum = args.detach_ymean
    else:
        model = GxE_Transformer(g_enc=g_enc,
                                e_enc=e_enc,
                                ld_enc=ld_enc,
                                gxe_enc=gxe_enc,
                                moe=moe,
                                config=config).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, y_scalers, env_scaler 

def main():
    args = parse_args()

    model_type = make_run_name(args)

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
    torch.cuda.set_device(0)
    print("Loading model...")
    model, y_scalers, env_scaler = load_model(device, args)    
    
    # load data
    print("Loading data...")
    test_data, test_loader = load_data(args,
                                       env_scaler=env_scaler,
                                       scaler=y_scalers,
                                       split="test",
                                       batch_size=32)

    # evaluate
    print("Evaluating model...")
    actuals, preds = evaluate(model, test_loader, y_scalers, device)
    plot_path = plot_results(model_type, actuals, preds)
    #save_results(model_type, actuals, preds)

    actuals = np.array(actuals)
    preds = np.array(preds).squeeze(-1)

    # TODO: log these to wandb
    pcc = pearsonr(actuals, preds).statistic
    mse = mean_squared_error(actuals, preds)
    print("Pearson Correlation:", pcc)
    print("Mean Squared Error:", mse)

    # log metrics
    run.summary["test/pearson"] = float(pcc)
    run.summary["test/mse"] = float(mse)
    run.summary["test/model_type"] = model_type
    
    run.log({
        "test/pearson": float(pcc),
        "test/mse": float(mse),
    })

    # plot image
    run.log({"test/pred_vs_actual": wandb.Image(str(plot_path))})
    run.finish()

if __name__ == "__main__":
    main()