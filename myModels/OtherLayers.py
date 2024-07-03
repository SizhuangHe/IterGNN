import torch
import torch.nn as nn
from torch.nn import LeakyReLU
from torch_geometric.nn import Linear
from torch_geometric.nn import MLP as _MLP

import numpy as np

def cal_size_list(in_channels, out_channels, layer_num):
    return np.linspace(
        in_channels, out_channels,
        layer_num+1, dtype='int'
    )

def MLP(size_list, last_activation=nn.LeakyReLU, activation=nn.LeakyReLU,
        last_bias=True, bias=True):
    last_bias = bias and last_bias
    return nn.Sequential(
        *(
            nn.Sequential(nn.Linear(size_list[ln], size_list[ln+1], bias=(bias if ln != len(size_list)-2 else last_bias)),
                           activation() if ln != len(size_list)-2 else last_activation())
            for ln in range(len(size_list)-1)
        )
    )

class VOCNodeEncoder(nn.Module):
    '''
    Node encoder by LRGB VOC-SP dataset
    '''
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = nn.Linear(14, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def reset_parameters(self):
        self.encoder.reset_parameters()

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super(GNNInductiveNodeHead, self).__init__()
        layers = []
        if num_layers > 1:
            layers.append(_MLP(in_channels=in_dim,
                                 hidden_channels=hid_dim,
                                 out_channels=hid_dim,
                                 num_layers=num_layers - 1,
                                 bias=True))
            layers.append(Linear(in_channels=hid_dim, out_channels=out_dim, bias=True))
        else:
            layers.append(Linear(in_channels=in_dim, out_channels=out_dim, bias=True))

        self.layer_post_mp = nn.Sequential(*layers)
                          
    def reset_parameters(self):
        for layer in self.layer_post_mp:
            layer.reset_parameters()       

    def forward(self, x):
        x = self.layer_post_mp(x)
        return x

class LobsterEncoder(nn.Module):
    def __init__(self, in_channel=int,
                 out_channel=int,
                 layer_num=int,
                 **kwargs):
        super(LobsterEncoder, self).__init__()
        # confidence_size_list = cal_size_list(hidden_size, 1, confidence_layer_num)
        module_size_list = cal_size_list(in_channel, out_channel, layer_num)
        self.module = MLP(module_size_list, **kwargs)

    def reset_parameters(self):
        for seq in self.module:
            for layer in seq:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()

    def forward(self, data):
        return self.module(data)