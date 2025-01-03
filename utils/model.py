import torch
import torch.nn as nn
import torch.nn.functional as F

# create transformer encoder for genotype data
class G_Encoder(nn.Module):

    def __init__(self,
                 output_dim: int = 768,
                 ):
        super().__init__()

    # forward pass
    def forward(self, x):
        return x

# create MLP encoder for environmental covariates
class E_Encoder(nn.Module):

    def __init__(self,
                 input_dim: int = 374,
                 output_dim: int = 768,
                 hidden_layer_sizes: list = [768, 768],
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.1,
                 batchnorm: bool = True,
                 ):
        super().__init__()

        # init hidden layers 
        for i, hidden_layer_size in enumerate(hidden_layer_sizes):

            # for first layer, go from GxE output to hidden layer size
            if i == 0:
                self.hidden_layers = [Block(input_dim, hidden_layer_size, activation, dropout, batchnorm)]    
            
            # for subsequent layers, go from previous hidden layer size to current hidden layer size
            else:
                self.hidden_layers.append(Block(hidden_layer_sizes[i-1], hidden_layer_size, activation, dropout, batchnorm))
        
        # add final layer
        self.final_layer = nn.Linear(hidden_layer_sizes[-1], output_dim)
        self.final_activation = activation

    # forward pass
    def forward(self, x):

        # through hidden layers
        for i, layer in enumerate(self.hidden_layers):

            # can't use residual connection for first layer, since input since doesn't match hidden layer sizes
            if i == 0:
                x = layer(x)
            else: 
                x = x + layer(x)
        
        # through final layer
        x = self.final_activation(self.final_layer(x))

        return x

class Block(nn.Module):
    """
    Flat block with a single linear layer, batchnorm, dropout and activation
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: nn.Module = nn.GELU(),
                 dropout: float = 0.1,
                 batchnorm: bool = True,
                 ):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(output_dim)
    
    # fwd pass
    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        if hasattr(self, 'batchnorm'):
            x = self.batchnorm(x)
        return x

class GPTransformer(nn.Module):

    """
    Full transformer for genomic prediction
    """
    
    def __init__(self,
                 dropout: float = 0.1,
                 hidden_layer_sizes: list = [768, 768],
                 hidden_activation: nn.Module = nn.GELU(),
                 final_activation: nn.Module = nn.Identity(),
                 g_enc: bool = True,
                 e_enc: bool = True,
                 ):
        super().__init__()

        # init G, E encoders
        if g_enc:
            self.g_encoder = G_Encoder()
        if e_enc:
            self.e_encoder = E_Encoder()

        # get output dimensions of G, E encoders (should be the same)
        if g_enc:
            self.gxe_output_dim = G_Encoder().output_dim
        else:
            self.gxe_output_dim = E_Encoder().output_dim

        # init fc layers
        for i, hidden_layer_size in enumerate(hidden_layer_sizes):

            # for first layer, go from GxE output to hidden layer size
            if i == 0:
                self.hidden_layers = [Block(self.gxe_output_dim, hidden_layer_size, hidden_activation, dropout)]    
            
            # for subsequent layers, go from previous hidden layer size to current hidden layer size
            else:
                self.hidden_layers.append(Block(hidden_layer_sizes[i-1], hidden_layer_size, hidden_activation, dropout))

        # init final layer (output of 1 for regression)
        self.final_layer = nn.Linear(768, 1) # CAN CHANGE INPUT, OUTPUT SIZE FOR LAYERS
        self.final_layer_activation = final_activation

    def forward(self, g, e):

        # only pass through G, E encoders if they exist
        if hasattr(self, 'g_encoder') and hasattr(self, 'e_encoder'):
            # pass through G, E encoders
            g_enc = self.g_encoder(x)
            e_enc = self.e_encoder(x)
            # concatenate encodings
            x = torch.cat([g_enc, e_enc], dim=1)

        elif hasattr(self, 'g_encoder'):
            # pass through G encoder
            x = self.g_encoder(x)

        elif hasattr(self, 'e_encoder'):
            # pass through E encoder
            x = self.e_encoder(x)

        # pass through other layers
        for layer in self.hidden_layers:
            # residual connection
            x = x + layer(x)

        # pass through final layer + activation
        x = self.final_layer_activation(self.final_layer(x))

        return x