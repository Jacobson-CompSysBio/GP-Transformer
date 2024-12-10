import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class E_Dataset(Dataset):
    """
    Dataset for E inputs only.
    Parameters:
        e_data (np.array): Environmental data
        target (np.array): Target variable
    Returns:
        e, target (tuple): Genomic data, Environmental data, Target variable
    """
    def __init__(self,
                 e_data,
                 target):
        
        # g, e should have the same length
        assert len(e_data) == len(target), "E and Target data should have the same number of rows"
        self.e_data = e_data
        self.target = target

    def __len__(self):
        return len(self.e_data)

    def __getitem__(self, idx):
        e = self.e_data[idx]
        target = self.target[idx]

        return e, target

class G_Dataset(Dataset):
    """
    Dataset for G inputs only.
    Parameters:
        g_data (np.array): Genomic data
        target (np.array): Target variable
    Returns:
        g, target (tuple): Genomic data, Environmental data, Target variable
    """
    def __init__(self,
                 g_data,
                 target,
                 tokenizer):
        
        # g, e should have the same length
        assert len(g_data) == len(target), "G and Target data should have the same number of rows"
        self.g_data = g_data
        self.target = target

        # tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.g_data)

    def __getitem__(self, idx):
        g = self.tokenizer(self.g_data[idx])
        target = self.target[idx]

        return g, target

class GxE_Dataset(Dataset):
    """
    Dataset for G and E inputs combined.
    Parameters:
        g_data (np.array): Genomic data
        e_data (np.array): Environmental data
        target (np.array): Target variable
    Returns:
        g, e, target (tuple): Genomic data, Environmental data, Target variable
    """
    def __init__(self,
                 g_data,
                 e_data,
                 target,
                 tokenizer):
        
        # g, e should have the same length
        assert len(g_data) == len(e_data) == len(target), "G, E, and Target data should have the same number of rows"
        self.g_data = g_data
        self.e_data = e_data
        self.target = target

        # tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.g_data)

    def __getitem__(self, idx):
        g = self.tokenizer(self.g_data[idx])
        e = self.e_data[idx]
        target = self.target[idx]

        return g, e, target