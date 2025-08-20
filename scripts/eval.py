# import packages 
import os, sys
from pathlib import Path
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

def load_data(
    model_type: str
) -> DataLoader:
    # load data
    if model_type == "e_model":
        test = E_Dataset(split='test', data_path = '../data/maize_data_2014-2022_vs_2023_v2/')
    elif model_type == "g_model":
        test = G_Dataset(split="test", data_path = '../data/maize_data_2014-2022_vs_2023_v2/')
    else:
        test = GxE_Dataset(split="test", data_path = '../data/maize_data_2014-2022_vs_2023_v2/')
    test_loader = DataLoader(test,
                            batch_size=64,
                            shuffle=True)
    return test_loader

def best_epoch(
    model_type: str
) -> int:
    with open(f'../logs/{model_type}/log.txt', 'r') as f:
        logs = f.readlines()
        logs = [l.strip() for l in logs][1:]
    logs = [l.split(',') for l in logs]
    logs = np.array(logs).astype(float)
    epochs, train_loss, val_loss = logs.T
    epochs = epochs.astype(int)
    epochs = np.arange(0, epochs.max() + 1, 1)
    best_epoch = int(epochs[np.argmin(val_loss)])
    return best_epoch

def load_model(
    model_type: str,
    best_epoch: int,
    device: torch.device
):
    # TODO: check for model type (GxE_Transformer / FullTransformer / LD)
    checkpoint_path = f'../checkpoints/best_weights/checkpoint_{best_epoch}.pt'
    # checkpoint_path = f'../checkpoints/{model_type}/checkpoint_{best_epoch}.pt'
    checkpoint = torch.load(checkpoint_path)["model"]
    if model_type == "e_model":
        model = GxE_Transformer(config=Config, g_enc=False).to(device)
    elif model_type == "g_model":
        model = GxE_Transformer(config=Config, e_enc=False).to(device)
    else:
        model = GxE_Transformer(config=Config).to(device)
    model.load_state_dict(checkpoint)
    return model

def eval(
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
                xb[key] = value.to(device)

            preds.extend(model(xb).detach().tolist())
            actuals.extend(yb.tolist())
    
    return actuals, preds

# TODO: what do with these?
pearsonr(actuals, preds)
mean_squared_error(actuals, preds)

def plot_results(
    model_type: str,
    actuals: list,
    preds: list
) -> None:
    #find line of best fit
    actuals = np.array(actuals).squeeze(-1)
    preds = np.array(preds).squeeze(-1)
    a, b = np.polyfit(actuals, preds, 1)

    #add points to plot
    plt.scatter(actuals, preds)

    #add line of best fit to plot
    plt.plot(actuals, a*actuals+b, color="orange")

    plt.title("Predicted vs. Actual Maize Yield")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.savefig(f"../data/results/{model_type}_pred_plot.png")

def save_results(
    model_type: str,
    actuals: list,
    preds: list
) -> None:
    # get locations
    locations = pd.read_csv('../data/maize_data_2014-2022_vs_2023_v2/location_2023.csv')
    location_names = list(np.unique(locations['Field_Location']))
    locations['actual'] = actuals
    locations['pred'] = preds

    location_results_df = pd.DataFrame({'location':[],
                                        'pearson':[]})
    for location in location_names:
        subset = locations[locations['Field_Location'] == location]
        pcc = pearsonr(subset['actual'], subset['pred'])[0]
        new_result = pd.DataFrame({'location': [location], 'pearson': [pcc]})
        location_results_df = pd.concat([location_results_df, new_result])
    location_results_df = location_results_df.reset_index(drop=True)
    location_results_df.to_csv(f'../data/results/{model_type}_location_results.csv')

def main():
    # TODO: make this a command line parameter
    model_type = "gxe_model"

    # load data
    test_loader = load_data(model_type)

    # load model
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    device = torch.device(f"cuda:{local_rank}")

    best_epoch = best_epoch(model_type)
    model = load_model(model_type, best_epoch, device)

    # evaluate
    actuals, preds = eval(device, model, test_loader)
    plot_results(model_type, actuals, preds)
    save_results(model_type, actuals, preds)

if __name__ == "__main__":
    main()