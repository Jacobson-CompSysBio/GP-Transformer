# import packages 
import os, sys
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

def load_data(
    model_type: str
) -> DataLoader:
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
    return test_loader

def load_model(
    model_type: str,
    device: torch.device
):
    # TODO: check for model type (GxE_Transformer / FullTransformer / LD)


    # get checkpoint with highest ending number
    #cpt_dir = os.listdir(f'checkpoints/')
    #best_cpt = max(cpt_dir, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(os.listdir('checkpoints/'))
    checkpoint = torch.load('checkpoints/checkpoint_0966.pt')["model"]

    # if model_type == "e_model":
    #     model = GxE_Transformer(config=Config, g_enc=False).to(device)
    # elif model_type == "g_model":
    #     model = GxE_Transformer(config=Config, e_enc=False).to(device)
    # else:
    #     model = GxE_Transformer(config=Config).to(device)

    # TODO: get config args from slurm script
    config = Config(block_size=2240,
                            n_layer=4,
                            n_head=16,
                            n_embd=768)
    model = GxE_LD_FullTransformer(config=config).to(device)
    model.load_state_dict(checkpoint)
    return model

def evaluate(
    model,
    test_loader: DataLoader,
    device: torch.device
) -> tuple[list, list]:
    preds = []
    actuals = []

    model.eval()
    with torch.no_grad():
    # loop through
        for xb, yb in tqdm(test_loader):
            # get things on device
            for key, value in xb.items():
                if isinstance(value, torch.Tensor):
                    if torch.isnan(value).any() or torch.isinf(value).any():
                        print(f"Invalid tensor found in input batch at key '{key}':")
                        print(f"  NaN values present: {torch.isnan(value).any().item()}")
                        print(f"  Inf values present: {torch.isinf(value).any().item()}")
                        continue
                    xb[key] = value.to(device, non_blocking=True)

            preds.extend(model(xb).detach().tolist())
            actuals.extend(yb['Yield_Mg_ha'])

    return actuals, preds

def plot_results(
    model_type: str,
    actuals: list,
    preds: list
) -> None:
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
    plt.savefig(f"data/results/{model_type}_pred_plot.png")

def save_results(
    model_type: str,
    actuals: list,
    preds: list
) -> None:
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

def main():
    # set up wand tracking
    # load_dotenv()
    # os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT")
    # os.environ["WANDB_ENTITY"] = os.getenv("WANDB_ENTITY")
    # os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

    # TODO: make this a command line parameter / environment variable (preferable)
    model_type = "gxe_model"

    # load data
    print("Loading data...")
    test_loader = load_data(model_type)

    # load model
    # local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    print("Loading model...")
    model = load_model(model_type, device)

    # evaluate
    print("Evaluating model...")
    actuals, preds = evaluate(model, test_loader, device)
    plot_results(model_type, actuals, preds)
    #save_results(model_type, actuals, preds)

    actuals = np.array(actuals)
    preds = np.array(preds).squeeze(-1)

    # TODO: log these to wandb
    print("Pearson Correlation:", pearsonr(actuals, preds))
    print("Mean Squared Error:", mean_squared_error(actuals, preds))

if __name__ == "__main__":
    main()