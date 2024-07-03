import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import JumpingKnowledge
import torch.nn.functional as F
import copy


class IterGNN(nn.Module):
    def __init__(self, 
                 gnn_layer_module=None, 
                 readout_module=None, 
                 confidence_module=None,
                 num_iter=1,
                 jk=None):
        assert(gnn_layer_module is not None and readout_module is not None and confidence_module is not None)
        super(IterGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module

        self.readout_module = readout_module
        self.confidence_module = confidence_module
        self.num_iter = num_iter
        if jk == 'cat':
            raise Exception("Cat JK is not supported by IterGNN due to varying number of iteration number! Use Max instead!")
        if jk is not None:
            self.jk = JumpingKnowledge(jk)
        else:
            self.jk = None
        

    def reset_parameters(self):
        self.gnn_layer_module.reset_parameters()
        self.confidence_module.reset_parameters()
        if self.jk is not None:
            self.jk.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr, batch):
        shape = x.size()
        new_x = x
        xs = [self.readout_module(Data(x=x, batch=batch))]
        left_confidence = torch.ones_like(x[:,0:1])
        for iter_num in range(self.num_iter):
            if torch.max(left_confidence).item() > 1e-7:
                data = Data(x=self.next_x(x, new_x, left_confidence, self.decreasing_ratio), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
                new_x = self.gnn_layer_module(data)
                global_feat = self.readout_module(data)
                current_confidence = self.confidence_module(global_feat)[batch]
                x = self.update_x(
                    x if iter_num != 0 else torch.zeros_like(x),
                    new_x, left_confidence, current_confidence, self.decreasing_ratio
                )
                left_confidence = self.update_confidence(left_confidence, current_confidence, self.decreasing_ratio)
                xs += [self.readout_module(Data(x=x, batch=batch))]
            else:
                break
        if self.jk is not None:
            x = self.jk(xs)

        x = x.reshape(shape)
        return x, iter_num
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio=1):
        return x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio=1):
        return left_confidence*(1.-current_confidence)
    @property
    def decreasing_ratio(self,):
        return None
    @staticmethod
    def next_x(x, new_x, left_confidence, decreasing_ratio=1):
        return new_x
    

class DeepGNN(nn.Module):
    def __init__(self, gnn_layer_module, readout_module, hid_dim, layer_num=1, jk=None, use_relu=True, use_bn=True):
        assert(gnn_layer_module is not None)
        super(DeepGNN, self).__init__()
        self.use_relu = use_relu
        self.use_bn = use_bn
        self.num_layers = layer_num
        self.layers = nn.ModuleList([copy.deepcopy(gnn_layer_module) for _ in range(layer_num)])
        if self.use_bn:
            self.batch_norms = nn.ModuleList(nn.BatchNorm1d(hid_dim) for _ in range(layer_num))
        self.readout_module = readout_module
        if jk is not None:
            self.jk = JumpingKnowledge(jk)
        else:
            self.jk = None

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        xs = [self.readout_module(Data(x=x, batch=batch))]
        for layer_idx in range(self.num_layers):
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            x = self.layers[layer_idx](data)
            if self.use_bn:
                x = self.batch_norms[layer_idx](x)
            if self.use_relu:
                x = F.relu(x)
            xs += [self.readout_module(Data(x=x, batch=batch))]
            assert(not torch.sum(torch.isnan(x)))
        
        if self.jk is not None:
            x = self.jk(xs)
        
        return x, len(self.layers)
    
def GNNArchitectures(gnn_layer_module=None, readout_module=None, hid_dim=220,confidence_module=None,
                     layer_name='IterGNN', num_layer_iter=1, jk=None, use_relu=True, use_bn=True):
    assert layer_name in ['IterGNN', 'DeepGNN']
    if layer_name == 'IterGNN':
        return IterGNN(gnn_layer_module, readout_module, confidence_module, num_iter=num_layer_iter, jk=jk)
    else:
        return DeepGNN(gnn_layer_module, readout_module, hid_dim, layer_num=num_layer_iter, jk=jk, use_relu=use_relu, use_bn=use_bn)