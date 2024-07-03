import torch
import torch.nn as nn

from torch_geometric.utils import softmax as gnn_softmax
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class ReadoutLayers(nn.Module):
    def __init__(self,
                 layer_name='Max'):
        super(ReadoutLayers, self).__init__()
        assert layer_name in ['Max', 'Mean', 'Sum']
        if layer_name == 'Max':
            self.pool =  global_max_pool
        elif layer_name == 'Mean':
            self.pool = global_mean_pool
        else:
            self.pool = global_add_pool

        
    def forward(self, data):
        x = data.x
        batch = data.batch
        return self.pool(x, batch)
