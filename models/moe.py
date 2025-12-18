# code modified from: https://apxml.com/posts/how-to-implement-moe-pytorch
import torch
from torch import nn
import torch.nn.functional as F

"""
Implementation of Mixture of Experts (MoE) layer for Genomic Prediction Transformer model.
"""

# Expert: individual 2-layer FFNs that "specialize" in something
class Expert(nn.Module):
    def __init__(self,
                 input_dim, 
                 hidden_dim,
                 output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# TopKGate: selects top K experts to compute the forward pass with
class TopKGate(nn.Module):
    def __init__(self,
                 input_dim,
                 num_experts,
                 k=2):
        super().__init__()
        self.k = k

        # linear layer computes logits for experts
        self.gate_linear = nn.Linear(input_dim, num_experts, bias=False)
    
    def forward(self, x):
        # x shape: [batch_size * seq_len, emb_dim]
        # logits shape: [batch_size * seq_len, num_experts]
        logits = self.gate_linear(x)

        # select topk experts
        # top_k_logits shape: [batch_size * seq_len, k]
        # top_k_indices shape: [batch_size * seq_len, k]
        top_k_logits, top_k_indices = torch.topk(logits, self.k, dim=-1)

        # apply softmax to topk logits for weighting between 0-1
        # top_k_weights shape: [batch_size * seq_len, k]
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # create a sparse weight matrix to combine outputs
        # full_weights shape: [batch_size * seq_len, num_experts]
        full_weights = torch.zeros_like(logits, dtype=top_k_weights.dtype)
        full_weights.scatter_(1, top_k_indices, top_k_weights)

        return full_weights, top_k_indices

# MoE layer: combines the gating network and experts  
class MoELayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_experts,
                 k=1,
                 expert_hidden_dim=None,
                 shared_expert: bool = False,
                 shared_expert_hidden_dim=None):
        super().__init__()
        if k < 1 or k > num_experts:
            raise ValueError(f"k must be in [1, num_experts] (got k={k}, num_experts={num_experts})")
        self.num_experts = num_experts
        self.k = k
        self.output_dim = output_dim

        # default to transformer embed size for MoE FFN width
        if expert_hidden_dim is None:
            expert_hidden_dim = input_dim * 4 # standard for MoE

        if shared_expert_hidden_dim is None:
            shared_expert_hidden_dim = expert_hidden_dim
        
        self.gate = TopKGate(input_dim, num_experts, k)
        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_hidden_dim, output_dim) for _ in range(num_experts)]
        )
        self.shared_expert = (
            Expert(input_dim, shared_expert_hidden_dim, output_dim)
            if shared_expert else None
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, emb_dim]
        original_shape = x.shape
        
        # flatten to [N, emb_dim] where N = batch_size * seq_len
        x = x.view(-1, original_shape[-1])

        # get gating weights and expert indices
        # gate_weights: [N, num_experts], top_k_indices: [N, k]
        gate_weights, top_k_indices = self.gate(x)

        # initialize final output tensor
        final_output = torch.zeros(x.shape[0], self.output_dim,
                                   device = x.device, dtype=x.dtype)
        
        # get indices for batch processing
        # flat_top_k_indices: [N * k]
        flat_top_k_indices = top_k_indices.view(-1)

        # map tokens to their assigned experts
        # create a flat tensor of inputs for batching across experts
        # flat_x: [N * k, emb_dim]
        flat_x = x.repeat_interleave(self.k, dim=0)

        # dispatch tokens to experts and compute outputs
        expert_outputs = []
        for i in range(self.num_experts):
            # find indices of tokens assigned to expert i
            # idx: [num_tokens_for_expert_i]
            idx = torch.where(flat_top_k_indices == i)[0]

            if idx.numel() > 0:
                # process tokens assigned to this expert
                expert_input = flat_x[idx]
                expert_output = self.experts[i](expert_input)

                # store output and original indices
                expert_outputs.append((idx, expert_output))
        
        # combine expert outputs using gating weights
        # we map results back to original token positions
        for idx, output in expert_outputs:
            # find the corresponding weights for these outputs
            # need original token indixes and expert indices
            original_indices = idx // self.k # get original token index (0 to N-1)
            expert_indices = flat_top_k_indices[idx] # which expert (0 to num_experts-1)

            # gather the weights using original and expert indices
            weights = gate_weights[original_indices, expert_indices].unsqueeze(1)

            # weight the expert output
            weighted_output = output * weights

            # add to the final output tensor at the correct positions
            # use index_add_ for scatter-add op.
            final_output.index_add(0, original_indices, weighted_output)

        # add shared expert output (always active)
        if self.shared_expert is not None:
            final_output = final_output + self.shared_expert(x)

        # reshape_back to original shape: [batch_size, seq_len, output_dim]
        final_output = final_output.view(original_shape[0], original_shape[1], self.output_dim)

        return final_output, gate_weights # return output, weights for aux loss
