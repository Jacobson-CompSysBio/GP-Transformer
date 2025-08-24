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

# load 
print("Loading data")
y_test = pd.read_csv('data/maize_data_2014-2023_vs_2024/y_test.csv')
X_test = pd.read_csv('data/maize_data_2014-2023_vs_2024/X_test.csv')

print(X_test['Yield_Mg_ha'])

## drop all rows in X_test that don't have a matching Env, Hybrid in Y_test
## and all rows in y_test that don't have a matching Env, Hybrid in X_test
#print("Filtering X_test to only include rows with matching Env, Hybrid in y_test")
#X_test = X_test[X_test[['Env', 'Hybrid']].apply(tuple, 1).isin(y_test[['Env', 'Hybrid']].apply(tuple, 1))]
#y_test = y_test[y_test[['Env', 'Hybrid']].apply(tuple, 1).isin(X_test[['Env', 'Hybrid']].apply(tuple, 1))]
#
## print number of rows in X_test
#print(f"Number of rows in X_test: {X_test.shape[0]}")
#print(f"Number of rows in y_test: {y_test.shape[0]}")
#
## save X_test
#X_test.to_csv('data/maize_data_2014-2023_vs_2024/X_test.csv')
#y_test.to_csv('data/maize_data_2014-2023_vs_2024/y_test.csv')