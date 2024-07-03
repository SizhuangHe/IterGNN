#!/usr/bin/env python
# coding=utf-8

import sys
import random
import numpy as np
from main.train import train_with_cross_val
from main.Planetoid_utils import add_noise, exp_per_model
from myModels.GNNLayers import GCNConv, GATConv, GINConv, SAGEConv
from myModels.GNNModel import NodeGNNModels, vanillaGNN_noisy
from myModels.ourModels import NodeIterativeGNN
from myModels.utils import make_uniform_schedule
from argparse import ArgumentParser
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GINConv
from torch_geometric.nn.models import MLP
from torch_geometric.transforms import NormalizeFeatures
from torch.nn.functional import cross_entropy
from torch_geometric.utils import add_remaining_self_loops
import wandb
wandb.login()

# you don't want to change information about the generated dataset in your sweep
parser = ArgumentParser()

parser.add_argument('-dir', '--data_dir', type=str, help="the path to the dataset", default="data/Planetoid")
parser.add_argument('-m', '--mode', type=str, choices=['Sweep', 'Run'], help="whether to initiate a wandb sweep or a single run", default='Run')
parser.add_argument('-dw', '--disable_wandb', action='store_true')

parser.add_argument("-d", "--dataset_name", type=str, choices=['Cora', 'CiteSeer', 'PubMed'],help="the dataset", default="Cora")
parser.add_argument('-np', '--noise_percent', type=float, help="noise percent", default=0)

parser.add_argument('-o', '--our_model', action='store_true')
parser.add_argument('-hd', '--hid_dim', type=int, help="the hidden dimension of the model", default=64)
parser.add_argument('-a', '--arc_name', choices=['IterGNN', 'DeepGNN'], help="the architecture of the GNN", default='IterGNN')
parser.add_argument('-ln', '--layer_num', type=int, 
                    help="if using a DeepGNN, this is the number of layers; if using our iterGNN, this is the number of iterations; if using their iterGNN, this is the MAX number of iterations", default=30)
parser.add_argument('-l', '--layer_name', type=str, choices=['GCNConv', 'GATConv', 'GINConv', 'SAGEConv'], help="the GNN layer to build up the model", default='GCNConv')
parser.add_argument('-r', '--readout_name', type=str, choices=['Max', 'Mean', 'Sum'], help="the name of the readout module", default='Max')
parser.add_argument('-hf', '--homo_flag',action='store_true')
parser.add_argument('-eln', '--encoder_layer_num', type=int, help="the number of layers in the encoder, for VOCSP, this is ignored and defaulted to 1", default=1)
parser.add_argument('-dln', '--decoder_layer_num', type=int, help="the number of layers in the decoder", default=1)
parser.add_argument('-gae', '--gnn_as_encoder', action='store_true', help="if set to true, will use a GNN layer as encoder instead of an MLP")
parser.add_argument('-bn', '--use_bn', action='store_true')
parser.add_argument('-rl', '--use_relu', action="store_true")
parser.add_argument('-jk', '--jk_type', type=str, choices=['cat', 'max'], help="integrate the jumpingknowledge model in the GNN", default=None)
parser.add_argument('--smooth_factor', type=float, default=0.7)
parser.add_argument('-lr', '--learning_rate', type=float, help="learning rate", default=0.001)
parser.add_argument('-wd', '--weight_decay', type=float, help="the weight decay", default=1e-5)
parser.add_argument('-ne', '--num_epochs', type=int, help="the number of epochs", default=100)
parser.add_argument('-mdo', '--mlp_dropout', type=float, help="the dropout rate of the final classification layers", default=0.5)
parser.add_argument('-gdo', '--gnn_dropout', type=float, help="the dropout rate of the GNN layers", default=0.5)

parser.add_argument('-s', '--seed', type=int, help="the random seed", default=-1)
args = parser.parse_args()



if args.mode == "Sweep":
    sweep_config = {
        'method': 'grid'
    }

    metric = {
        'name': 'accuracy',
        'goal': 'maximize'
    }
    sweep_config['metric'] = metric

    parameters_dict = {
        'num_iter_layers': {
            'value': args.layer_num
        },
        'learning_rate': {
            'values': [0.00001, 0.00003, 0.00005, 0.00007, 0.0001, 0.0003, 0.0005, 0.0007, 0.001]
        },
        'smooth_fac': {
            'values': [0.6, 0.7, 0.8]
        },
        'hid_dim': {
            'value': args.hid_dim
        },
        'weight_decay': {
            'values': [0, 1e-5, 1e-4]
        },
        'num_epochs': {
            'value': args.num_epochs
        },
        'mlp_dropout': {
            'value': args.mlp_dropout
        },
        'gnn_dropout': {
            'value': args.gnn_dropout
        },
        'dataset_name': {
            'value': args.dataset_name
        },
        'layer_name': {
            'value': args.layer_name
        },
        'readout_name': {
            'value': args.readout_name
        },
        'homo_flag': {
            'value': args.homo_flag
        },
        'arc_name':{
            'value': args.arc_name
        },
        'encoder_layer_num':{
            'value':args.encoder_layer_num
        },
        'decoder_layer_num':{
            'value':args.decoder_layer_num
        },
        'pct_start':{
            'value': 0.2
        },
        'lr_scheduler':{
            'value': 'OneCyleLR'
        },
        'jk':{
            'value':args.jk_type
        },
        "noise_percent":{
            'value':args.noise_percent
        },
        "seed":{
            'value':args.seed
        },
        'use_relu':{
            'value': args.use_relu
        },
        'use_bn':{
            "value": args.use_bn
        },
        'our_model':{
            "value": args.our_model
        }
    }
    sweep_config['parameters'] = parameters_dict
else:
    run_config = {
        'num_iter_layers': args.layer_num,
        'learning_rate': args.learning_rate,
        'smooth_fac': args.smooth_factor,
        'hid_dim':args.hid_dim,
        'weight_decay':args.weight_decay,
        'num_epochs':args.num_epochs,
        'mlp_dropout':args.mlp_dropout,
        'gnn_dropout':args.gnn_dropout,
        'dataset_name':args.dataset_name,
        'layer_name':args.layer_name,
        'readout_name':args.readout_name,
        'homo_flag':args.homo_flag,
        'arc_name':args.arc_name,
        'encoder_layer_num':args.encoder_layer_num,
        'decoder_layer_num':args.decoder_layer_num,
        'pct_start': 0.2,
        'lr_scheduler':'OneCycleLR',
        'jk': args.jk_type,
        'noise_percent':args.noise_percent,
        'seed': args.seed,
        "use_relu": args.use_relu,
        'use_bn': args.use_bn,
        "our_model": args.our_model
    }

if args.disable_wandb:
    wandb_mode = 'disabled'
else:
    wandb_mode = None

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterGNN", 
               config=config, 
               notes="Sweep for the IterGNN, from McCleary",
               tags=["IterGNN"],
               dir="/vast/palmer/scratch/dijk/sh2748",
               mode=wandb_mode)
    config = wandb.config
    print(str(config))
    dataset = Planetoid(root=args.data_dir, name=args.dataset_name, transform=NormalizeFeatures())
    data = dataset[0]
    data.edge_index = add_remaining_self_loops(data.edge_index)[0]
    data = add_noise(data, percent=config.noise_percent, seed=2147483647)
    in_channel, out_channel = dataset.num_features, dataset.num_classes
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)

    if config.seed >0:
    # pass in -1 as seed during sweep
        seed = config.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print("+++ Random seed: {} +++".format(seed))
        print()
    
    if config.arc_name == "IterGNN":
        layer_num = config.num_iter_layers
        encoder = MLP(
                    in_channels=in_channel,
                    hidden_channels=config.hid_dim,
                    out_channels=config.hid_dim,
                    num_layers=config.encoder_layer_num
                    )
        decoder = MLP(
                        in_channels=config.hid_dim,
                        hidden_channels=config.hid_dim,
                        out_channels=out_channel,
                        num_layers=config.decoder_layer_num
                    )
    elif config.arc_name == "DeepGNN":
        layer_num = config.num_iter_layers - 2 # In this case, the encoder and decoder also count as one layer
        
        if config.layer_name == 'GCNConv':
            encoder = GCNConv(in_channel=in_channel, out_channel=config.hid_dim, homogeneous_flag=config.homo_flag)
            decoder = GCNConv(in_channel=config.hid_dim, out_channel=out_channel, homogeneous_flag=config.homo_flag)
        elif config.layer_name == 'GATConv':
            encoder = GATConv(in_channel=in_channel, out_channel=config.hid_dim, homogeneous_flag=config.homo_flag)
            decoder = GATConv(in_channel=config.hid_dim, out_channel=out_channel, homogeneous_flag=config.homo_flag)
        elif config.layer_name == 'GINConv':
            raise NotImplementedError
        elif config.layer_name == 'SAGEConv':
            encoder = SAGEConv(in_channel=in_channel, out_channel=config.hid_dim, homogeneous_flag=config.homo_flag)
            decoder = SAGEConv(in_channel=config.hid_dim, out_channel=out_channel, homogeneous_flag=config.homo_flag)
    else:
        raise NotImplementedError
    
    

    if config.our_model:
        assert config.arc_name == "IterGNN"
        net = NodeIterativeGNN(layer_name=config.layer_name,
                           hidden_size=config.hid_dim,
                           train_schedule=train_schedule,
                           homogeneous_flag=config.homo_flag,
                           mlp_dropout=config.mlp_dropout,
                           gnn_dropout=config.gnn_dropout,
                           readout_name=config.readout_name,
                           encoder=encoder,
                           decoder=decoder,
                           jk=config.jk,
                           use_bn=config.use_bn,
                           use_relu=config.use_relu
                           )
    else:
        if config.arc_name == "IterGNN":
            net = NodeGNNModels(architecture_name=config.arc_name,
                            layer_num=layer_num,
                            layer_name=config.layer_name,
                            hidden_size=config.hid_dim,
                            readout_name=config.readout_name,
                            homogeneous_flag=config.homo_flag,
                            confidence_layer_num=1,
                            encoder=encoder,
                            decoder=decoder,
                            jk=config.jk,
                            use_bn=config.use_bn,
                            use_relu=config.use_relu
                            )
        else:
            net = vanillaGNN_noisy(architecture_name=config.arc_name,
                            layer_num=layer_num,
                            layer_name=config.layer_name,
                            hidden_size=config.hid_dim,
                            readout_name=config.readout_name,
                            homogeneous_flag=config.homo_flag,
                            confidence_layer_num=1,
                            encoder=encoder,
                            decoder=decoder,
                            jk=config.jk,
                            use_bn=config.use_bn,
                            use_relu=config.use_relu)

    print(str(net))
    net = net.to(device)
    optimizer= torch.optim.Adam(net.parameters(), lr = config.learning_rate, weight_decay=config.weight_decay)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=config.num_epochs)

    exp_per_model(net, data, optimizer, scheduler=None, config=config, device=device)
    


    
    

if args.mode == 'Sweep':
    sweep_id = wandb.sweep(sweep_config, project="IterGNN")
    wandb.agent(sweep_id, run_exp)
else:
    run_exp(run_config)

