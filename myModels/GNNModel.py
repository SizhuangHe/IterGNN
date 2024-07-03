import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.models import MLP
from torch_geometric.nn import JumpingKnowledge
from .OtherLayers import cal_size_list, MLP, VOCNodeEncoder, GNNInductiveNodeHead, LobsterEncoder
from .GNNLayers import GNNLayer, GCNConv, GINConv, GATConv, SAGEConv
from .GNNAgg import ReadoutLayers
from .GNNArch import GNNArchitectures

class _GNNModels(nn.Module):
    def __init__(self,
                 architecture_name='IterGNN', layer_num=10,
                 layer_name='GCNConv', hidden_size=64,
                 readout_name='Max',
                 homogeneous_flag=0,
                 confidence_layer_num=1,
                 encoder=None,
                 decoder=None,
                 jk=None,
                 use_relu = True,
                 use_bn=True):
        super(_GNNModels, self).__init__()
        self.jk = jk
        self.embedding_module = encoder
        self.head_module = decoder
        self.readout_module = ReadoutLayers(layer_name=readout_name)
        
        gnn_layer = GNNLayer(layer_name=layer_name, in_channel=hidden_size, out_channel=hidden_size, homogeneous_flag=homogeneous_flag)
        self.confidence_module = LobsterEncoder(hidden_size, 1, confidence_layer_num, last_activation=nn.Sigmoid)
        self.gnn_module = GNNArchitectures(gnn_layer_module=gnn_layer,
                                        readout_module=self.readout_module,
                                        hid_dim=hidden_size,
                                        confidence_module=self.confidence_module,
                                        layer_name=architecture_name,
                                        num_layer_iter=layer_num,
                                        jk=self.jk,
                                        use_relu=use_relu,
                                        use_bn=use_bn
                                        )   
    def readout(self, data):
        raise NotImplementedError
        # return self.readout_module(data)
    
    def reset_parameters(self):
        self.embedding_module.reset_parameters()
        self.head_module.reset_parameters()
        self.confidence_module.reset_parameters()
        self.gnn_module.reset_parameters()


    def forward(self, data):
        x, batch, edge_index, edge_attr = data.x, data.batch, data.edge_index, data.edge_attr

        x = self.embedding_module(x)

        global_feat, cur_layer_num = self.gnn_module(x, edge_index, edge_attr, batch)
        
        if self.jk is None:
            data = Data(x=global_feat, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
            global_feat = self.readout(data)
            
        output = self.head_module(global_feat)
        return output, cur_layer_num
    @property
    def tao(self,):
        return self.gnn_module_list[0].tao

class GraphGNNModels(_GNNModels):
    def readout(self, data):
        return self.readout_module(data)

class NodeGNNModels(_GNNModels):
    def readout(self, data):
        return data.x

class vanillaGNN_noisy(_GNNModels):
    def forward(self, data):
        x, batch, edge_index, edge_attr = data.x, data.batch, data.edge_index, data.edge_attr
        x = self.embedding_module(x, edge_index)
        global_feat, cur_layer_num = self.gnn_module(x, edge_index, edge_attr, batch)
        output = self.head_module(global_feat, edge_index)
        return output, cur_layer_num
