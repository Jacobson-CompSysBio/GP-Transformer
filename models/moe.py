import torch
from torch import nn
import torch.nn.functional as F

"""
Implementation of Mixture of Experts (MoE) layer for Genomic Prediction Transformer model.
"""

# the expert module: "specializes" on certain types of inputs
class Expert(nn.Module):
    """
    the expert is just a simple 2-layer ffnn
    input --> relu --> output
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# the gating network: controls which experts are active
class GatingNetwork(nn.Module):
    """
    The gate is just a linear layer with output size <num_experts>.
    We apply softmax to get the probability distribution (of activity) over experts.
    """
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        return F.softmax(self.gate(x), dim=2)

# the mixture of experts layer: combines the expert module and the gating network with a transformer. 
# instead of a typical dense layer following self-attention in the transformer block, we replace with an MoE.
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoELayer, self).__init__()
        
        # we initialize <num_experts> experts and a single gate to control routing
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.gate = GatingNetwork(input_dim, num_experts)
    
    def forward(self, x, num_experts_per_tok):
        pass

