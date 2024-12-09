# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoModelForMaskedLM, 
    AutoTokenizer
    )
from transformers.models.bert.configuration_bert import BertConfig

# ----------------------------------------------------------------
# build the model

class MLP(nn.Module):
    """
    MLP Block for GPTransformer
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        """
        Parameters:
            dim (int): input dimension
            dropout (float): dropout rate
        """
        super().__init__()

        # keeping the mlp structure from GPT
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
    
    def forward(self, x):
        x = self.net(x)
        return x

class GPTransformer(nn.Module):
    
    """
    Transformer model for multi-task genomic prediction. Inherits from DNA-BERT 2 (https://github.com/MAGICS-LAB/DNABERT_2)
    """

    def __init__(self,
                 model_path: str,
                 dropout: float = 0.1,
                 n_mlp: int = 1,
                 n_out: int = 4,
                 model_type: str = 'regression'):
        
        """
        Parameters:
            model_path (str): path to the pre-trained model
            dropout (float): dropout rate
            n_mlp (int): number of MLP blocks
            n_out (int): number of output classes
        """
        
        # model type must be either 'regression' or 'classification'
        assert model_type in ['regression', 'classification'], f"model_type must be either 'regression' or 'classification', got {model_type}"
        super().__init__()

        # user-defined hyper-parameters
        self.model_name = model_path.split('/')[-1]
        self.dropout = dropout
        self.n_mlp = n_mlp
        self.n_out = n_out
        self.model_type = model_type

        # init hf model
        config = BertConfig.from_pretrained(f"{model_path}")
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(f'{model_path}', 
                                                                           trust_remote_code=True,
                                                                           config=config)

        # get embedding size from model
        self.n_embed = self.hf_model.config.hidden_size

        # create MLP block
        if n_mlp > 0:
            self.mlp_layers = nn.ModuleList([MLP(self.n_embed, self.dropout) for _ in range(self.n_mlp)])

        # create output layer
        self.out = nn.Linear(self.n_embed, self.n_out)

    def forward(self, input_ids, attention_mask):

        # pass through pre-trained BERT model
        # THIS LINE IS MODEL-DEPENDENT - NEEDS TO BE CHANGED FOR OTHER MODELS
        x = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)[1][:, 0, :] # only taken first hidden state output ([CLS token])

        # pass through MLP blocks
        if self.n_mlp > 0:
            for mlp in self.mlp_layers:
                x = x + mlp(x)

        # pass through output layer
        x = self.out(x)

        return x

# ----------------------------------------------------------------



