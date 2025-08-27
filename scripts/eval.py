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

import random
random.seed(0); np.random.seed(0); torch.manual_seed(0)

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data(model_type: str):

    # load data
    if model_type == "e_model":
        test = E_Dataset(split='test', data_path = 'data/maize_data_2014-2023_vs_2024/')
    elif model_type == "g_model":
        test = G_Dataset(split="test", data_path = 'data/maize_data_2014-2023_vs_2024/')
    else:
        test = GxE_Dataset(split="test", data_path = 'data/maize_data_2014-2023_vs_2024/')
    test_loader = DataLoader(test,
                            batch_size=32,
                            shuffle=False,
                            )
    return test, test_loader

def evaluate(model,
             test_loader: DataLoader,
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

            preds.extend(model(xb).detach().tolist())
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--g_enc", type=str2bool, default=True)
    p.add_argument("--e_enc", type=str2bool, default=True)
    p.add_argument("--ld_enc", type=str2bool, default=True)
    p.add_argument("--final_tf", type=str2bool, default=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--early_stop", type=int, default=50)
    p.add_argument("--layers_per_block", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--emb_size", type=int, default=768)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--loss", type=str, default="mse",
                   choices=["mse", "pcc", "both"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight for MSE when --loss both (loss = alpha*MSE + (1-alpha)*(1-PCC))")
    p.add_argument('--checkpoint_dir', type=str, required=True,     # optional manual override
                   help='Directory from train.py for this run')
    return p.parse_args()

def make_run_name(args) -> str:
    g = "g+" if args.g_enc else ""
    e = "e+" if args.e_enc else ""
    ld = "ld+" if args.ld_enc else ""
    tf = "tf+" if args.final_tf else ""
    model_type = g + e + ld + tf
    model_type = model_type[:-1]
    loss_tag = args.loss
    return (
        f"{model_type}_{loss_tag}_{args.batch_size}bs_{args.lr}lr_{args.weight_decay}wd_"
        f"{args.num_epochs}epochs_{args.early_stop}es_{args.layers_per_block}lpb_"
        f"{args.heads}heads_{args.emb_size}emb_{args.dropout}do"    
    )

def load_model(dataset: Dataset,
               device: torch.device,
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
    final_tf = config.get("final_tf", args.final_tf)
    blk = config.get("block_size", len(dataset[0][0]['g_data']))
    n_layer = config.get("n_layer", args.layers_per_block)
    n_head = config.get("n_head", args.heads)
    n_embd = config.get("n_embd", args.emb_size)
    loss = config.get("loss", args.loss)
    alpha = config.get("alpha", args.alpha)

    config = Config(block_size=blk,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_embd=n_embd)
    model = GxE_Transformer(g_enc=g_enc,
                            e_enc=e_enc,
                            ld_enc=ld_enc,
                            final_tf=final_tf,
                            config=config).to(device)

    model.load_state_dict(state, strict=False)
    model.eval()

    return model

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

    # load data
    print("Loading data...")
    test_data, test_loader = load_data(model_type)

    # load model
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    print("Loading model...")
    model = load_model(test_data,
                       device,
                       args)

    # evaluate
    print("Evaluating model...")
    actuals, preds = evaluate(model, test_loader, device)
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