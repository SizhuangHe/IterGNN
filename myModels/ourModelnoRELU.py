import torch.nn as nn
from torch_geometric.data import Data
from .GNNLayers import GNNLayer
from .OtherLayers import cal_size_list, MLP
from .GNNAgg import ReadoutLayers
import torch.nn.functional as F

class iterArch(nn.Module):
    def __init__(self, gnn_layer, train_schedule, eval_schedule, dropout, xavier_init=False):
        super(iterArch, self).__init__()
        self.layer = gnn_layer
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()
    
    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # GCNConv layers are already Xavier initilized
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
      
    def _next_x(self, old_x, new_x, smooth_fac):
        next_x = smooth_fac * old_x + (1 - smooth_fac) * new_x
        return next_x

    def forward(self, x, edge_index, batch):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        
        for smooth_fac in schedule:      
            old_x = x
            data = Data(x=x, edge_index=edge_index)
            x = self.layer(data)
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac) 
        
        return x




class iterativeGNN(nn.Module):
    def __init__(self, in_channel=int,
                 out_channel=int,
                 embedding_layer_num=2,
                 layer_name='GCNConv', hidden_size=64,
                 train_schedule=None,
                 eval_schedule=None,
                 homogeneous_flag=0,
                 dropout=0,
                 readout_name='Max',
                 head_layer_num=1,):
        super(iterativeGNN, self).__init__()
        self.num_iter = len(train_schedule)
        
        embedding_size_list = cal_size_list(in_channel, hidden_size, embedding_layer_num)
        head_size_list = cal_size_list(hidden_size, out_channel, head_layer_num)
        
        self.embedding_module = MLP(embedding_size_list)
        self.head_module = MLP(head_size_list, activation=nn.Identity)
        self.readout_module = ReadoutLayers(layer_name=readout_name)

        gnn_layer = GNNLayer(layer_name=layer_name, in_channel=hidden_size, out_channel=hidden_size, homogeneous_flag=homogeneous_flag)
        self.gnn_module = iterArch(gnn_layer, train_schedule, eval_schedule,dropout)

    def readout(self, data):
        return self.readout_module(data)
    
    def forward(self, data):
        x, batch, edge_index = data.x, data.batch, data.edge_index

        x = self.embedding_module(x)

        x= self.gnn_module(x, edge_index, batch)
        data = Data(x=x, edge_index=edge_index, batch=data.batch)
        global_feat = self.readout(data)
        output = self.head_module(global_feat)
        
        return output, self.num_iter