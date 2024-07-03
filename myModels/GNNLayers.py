import torch
import torch.nn as nn

from torch_geometric.nn import GCNConv as _GCNConv, GATConv as _GATConv, GINConv as _GINConv, SAGEConv as _SAGEConv
from torch_geometric.nn.models import MLP

from .utils import homo_gnn_softmax
import torch.nn.functional as F

class _HomoGATConv(_GATConv):
    """
    Configed GATConv from torch_geometric.nn (1.3.2)
    Single change: softmax --> homo_gnn_softmax
    """
    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        assert(not torch.sum(torch.isnan(x_i)))
        assert(not torch.sum(torch.isnan(x_j)))
        assert(not torch.sum(torch.isnan(edge_index_i)))
        x_j = x_j.view(-1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        assert(not torch.sum(torch.isnan(alpha)))
        alpha = F.leaky_relu(alpha, self.negative_slope)
        assert(not torch.sum(torch.isnan(alpha)))
        alpha = homo_gnn_softmax(alpha, edge_index_i, size_i)
        assert(not torch.sum(torch.isnan(alpha)))

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

class GCNConv(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=0, homogeneous_flag=False):
        super(GCNConv, self).__init__()
        self.module = _GCNConv(in_channel, out_channel, bias=not homogeneous_flag)

    def reset_parameters(self):
        self.module.reset_parameters()
    
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class GATConv(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=0, homogeneous_flag=False):
        super(GATConv, self).__init__()
        if not homogeneous_flag:
            self.module = _GATConv(in_channel, out_channel)
        else:
            self.module = _HomoGATConv(in_channel, out_channel, bias=False)

    def reset_parameters(self):
        self.module.reset_parameters()
    
    def forward(self, x, edge_index):
        return self.module(x, edge_index)
    
class GINConv(nn.Module):
    def __init__(self, in_channel, out_channel, homogeneous_flag=False, num_layers=5):
        super(GINConv, self).__init__() 
        mlp = MLP(in_channels= in_channel, hidden_channels=in_channel, out_channels=out_channel, num_layers = num_layers, bias= not homogeneous_flag)
        self.module = _GINConv(mlp)
    
    def reset_parameters(self):
        self.module.reset_parameters()

    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class SAGEConv(nn.Module):
    '''
    We use a MAX aggregator and normalize hidden representations for SAGEConv.
    '''
    def __init__(self, in_channel, out_channel, aggr="max", normalize=True, root_weight=True, project=False, homogeneous_flag = False):
        super(SAGEConv, self).__init__()
        print("aggr:", aggr)
        self.module = _SAGEConv(in_channel, out_channel, aggr, normalize, root_weight, project, bias=not homogeneous_flag)
    def reset_parameters(self):
        self.module.reset_parameters()
    
    def forward(self, x, edge_index):
        return self.module(x, edge_index)

class GNNLayer(nn.Module):
    def __init__(self, 
                 layer_name=str,
                 in_channel=int,
                 out_channel=int,
                 homogeneous_flag=False,
                 num_mlp_layers=5):
        super(GNNLayer, self).__init__()
        assert layer_name in ["GCNConv", "GATConv", "GINConv", "SAGEConv"]
        if layer_name == "GCNConv":
            self.layer = GCNConv(in_channel=in_channel, out_channel=out_channel, homogeneous_flag=homogeneous_flag)
        elif layer_name == "GATConv":
            self.layer = GATConv(in_channel=in_channel, out_channel=out_channel, homogeneous_flag=homogeneous_flag)
        elif layer_name == "GINConv":
            self.layer = GINConv(in_channel=in_channel, out_channel=out_channel, homogeneous_flag=homogeneous_flag, num_layers=num_mlp_layers)
        else:
            print("lala")
            self.layer = SAGEConv(in_channel, out_channel, homogeneous_flag=homogeneous_flag)
            

    def reset_parameters(self):
        self.layer.reset_parameters()
    
    def forward(self, data):
        return self.layer(data.x, data.edge_index)