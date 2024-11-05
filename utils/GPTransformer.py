# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForMaskedLM, AutoTokenizer

class GPTransformer(nn.Module):
    """
    Transformer model for multi-task genomic prediction. Inherits from DNA-BERT 2 (https://github.com/MAGICS-LAB/DNABERT_2)
    """

    def __init__(self,
                 model_name: str):
        
        super().__init__()

        self.model_name = model_name
        self.model = AutoModelForMaskedLM.from_pretrained(f'zhangtaolab/{model_name}', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(f'zhangtaolab/{model_name}', trust_remote_code=True)

        # 
        self.fc_layer = nn.Linear()
