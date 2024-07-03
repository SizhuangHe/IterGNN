import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import JumpingKnowledge
from .GNNLayers import GNNLayer
from .OtherLayers import cal_size_list, MLP
from .GNNAgg import ReadoutLayers
import torch.nn.functional as F
from torch.nn import BatchNorm1d

class iterArch(nn.Module):
    def __init__(self, gnn_layer, readout,train_schedule, eval_schedule, dropout, hid_dim, use_bn,use_relu=True,jk=None, xavier_init=False,):
        super(iterArch, self).__init__()
        self.layer = gnn_layer
        self.readout = readout
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.batch_norm = BatchNorm1d(hid_dim)
        self.train_schedule = train_schedule
        if eval_schedule is not None:
            self.eval_schedule = eval_schedule
        else:
            self.eval_schedule = self.train_schedule
        self.dropout = dropout
        if xavier_init:
            self._init_xavier()
        if jk is not None:
            self.jk = JumpingKnowledge(jk)
        else:
            self.jk = None
    
    def reset_parameters(self):
        self.layer.reset_parameters()
        if self.jk is not None:
            self.jk.reset_parameters()

    def _init_xavier(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): # GCNConv layers are already Xavier initilized
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
      
    def _next_x(self, old_x, new_x, smooth_fac):
        next_x = smooth_fac * old_x + (1 - smooth_fac) * new_x
        return next_x

    def forward(self, x, edge_index, edge_attr, batch):
        if self.training:
            schedule = self.train_schedule
        else:
            schedule = self.eval_schedule
        xs = [self.readout(Data(x=x, batch=batch))]
        for smooth_fac in schedule:      
            old_x = x
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            x = self.layer(data)
            if self.use_bn:
                x = self.batch_norm(x)
            if self.use_relu:
                x = F.relu(x)
            new_x = F.dropout(x, self.dropout, training=self.training)
            x = self._next_x(old_x, new_x, smooth_fac)
            
            xs += [self.readout(Data(x=x, batch=batch))]
        
        if self.jk is not None:
            x = self.jk(xs)
        
        return x




class IterativeGNN(nn.Module):
    def __init__(self,
                 layer_name='GCNConv', hidden_size=64,
                 train_schedule=None,
                 eval_schedule=None,
                 homogeneous_flag=0,
                 mlp_dropout=0,
                 gnn_dropout=0,
                 use_bn=True,
                 jk=None,
                 readout_name='Max',
                 encoder=None,
                 decoder=None,
                 use_relu=True
                 ):
        super(IterativeGNN, self).__init__()
        self.num_iter = len(train_schedule)
        self.mlp_dropout = mlp_dropout
        self.jk=jk
        
        
        self.embedding_module = encoder
        self.head_module = decoder
        self.readout_module = ReadoutLayers(layer_name=readout_name)

        gnn_layer = GNNLayer(layer_name=layer_name, in_channel=hidden_size, out_channel=hidden_size, homogeneous_flag=homogeneous_flag)
        self.gnn_module = iterArch(gnn_layer=gnn_layer, readout=self.readout_module, train_schedule=train_schedule, eval_schedule=eval_schedule,dropout=gnn_dropout, hid_dim=hidden_size,use_bn=use_bn, use_relu=use_relu,jk=jk,)

    def reset_parameters(self):
        self.embedding_module.reset_parameters()
        self.head_module.reset_parameters()
        self.gnn_module.reset_parameters()

    def readout(self, data):
        raise NotImplementedError
    
    def forward(self, data):
        x, batch, edge_index, edge_attr = data.x, data.batch, data.edge_index, data.edge_attr

        x = self.embedding_module(x)

        global_feat= self.gnn_module(x, edge_index, edge_attr, batch)
        if self.jk is None:
            data = Data(x=global_feat, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            global_feat = self.readout(data)
        global_feat = F.dropout(global_feat, p=self.mlp_dropout, training=self.training)
        output = self.head_module(global_feat)
        
        return output, self.num_iter
    
class GraphIterativeGNN(IterativeGNN):
    def readout(self, data):
        return self.readout_module(data)
        
class NodeIterativeGNN(IterativeGNN):
    def readout(self, data):
        return data.x   