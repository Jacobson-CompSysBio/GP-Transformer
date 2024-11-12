# import necessary libraries
import time
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoModelForMaskedLM, 
    AutoTokenizer
    )

from transformers.models.bert.configuration_bert import BertConfig

# ----------------------------------------------------------------

class SNPDataset(Dataset):

    def __init__(self, data, tokenizer):
        # NOTE: not sure what our input data will look like yet... 
        # probably want to make it a numpy array or tensor or something
        

        pass

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.dataset)
    
    def __getitem__(self, index):
        """
        Parameters
            index (int): index to return data from
        Returns:
            tokens (torch.tensor): tokenized SNP sequence
            attention_mask (torch.tensor): attention mask for tokenized SNP sequence
        """


        return tokens, attention_mask