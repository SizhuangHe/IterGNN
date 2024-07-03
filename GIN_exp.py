#!/usr/bin/env python
# coding=utf-8

import sys

from main.train import train_with_cross_val
from myModels.GNNModel import GraphGNNModels
from myModels.ourModels import GraphIterativeGNN
from myModels.utils import make_uniform_schedule
from argparse import ArgumentParser
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv
from torch_geometric.nn.models import MLP
from torch_geometric.transforms import OneHotDegree
from torch.nn.functional import cross_entropy
import wandb
wandb.login()

# you don't want to change information about the generated dataset in your sweep
parser = ArgumentParser()

parser.add_argument('-dir', '--data_dir', type=str, help="the path to the dataset", default="/vast/palmer/scratch/dijk/sh2748/data/TUDataset")
parser.add_argument('-m', '--mode', type=str, choices=['Sweep', 'Run'], help="whether to initiate a wandb sweep or a single run", default='Run')
parser.add_argument('-dw', '--disable_wandb', action='store_true')

parser.add_argument("-d", "--dataset_name", type=str, choices=['IMDBB', 'IMDBM', 'PROTEINS', 'MUTAG'],help="the dataset", default="IMDBB")
parser.add_argument('-bs', '--batch_size', type=int, help="the batch size for mini-batch", default=32)

parser.add_argument('-o', '--our_model', action='store_true')
parser.add_argument('-hd', '--hid_dim', type=int, help="the hidden dimension of the model", default=64)
parser.add_argument('-a', '--arc_name', choices=['IterGNN', 'DeepGNN'], help="the architecture of the GNN", default='IterGNN')
parser.add_argument('-ln', '--layer_num', type=int, 
                    help="if using a DeepGNN, this is the number of layers; if using our iterGNN, this is the number of iterations; if using their iterGNN, this is the MAX number of iterations", default=30)
parser.add_argument('-l', '--layer_name', type=str, choices=['GCNConv', 'GATConv', 'GINConv'], help="the GNN layer to build up the model", default='GCNConv')
parser.add_argument('-r', '--readout_name', type=str, choices=['Max', 'Mean', 'Sum'], help="the name of the readout module", default='Max')
parser.add_argument('-hf', '--homo_flag',action='store_true')
parser.add_argument('-eln', '--encoder_layer_num', type=int, help="the number of layers in the encoder, for VOCSP, this is ignored and defaulted to 1", default=1)
parser.add_argument('-dln', '--decoder_layer_num', type=int, help="the number of layers in the decoder", default=3)
parser.add_argument('-gae', '--gnn_as_encoder', action='store_true', help="if set to true, will use a GNN layer as encoder instead of an MLP")
parser.add_argument('-bn', '--use_bn', action='store_true')
parser.add_argument('-jk', '--jk_type', type=str, choices=['cat', 'max'], help="integrate the jumpingknowledge model in the GNN", default=None)

parser.add_argument('-nf', '--num_folds', type=int, help="the number K in K-folds cross validation", default=10)
parser.add_argument('-lr', '--learning_rate', type=float, help="learning rate", default=0.001)
parser.add_argument('-wd', '--weight_decay', type=float, help="the weight decay", default=1e-5)
parser.add_argument('-ne', '--num_epochs', type=int, help="the number of epochs", default=100)
parser.add_argument('-mdo', '--mlp_dropout', type=float, help="the dropout rate of the final classification layers", default=0.5)
parser.add_argument('-gdo', '--gnn_dropout', type=float, help="the dropout rate of the GNN layers", default=0.5)

parser.add_argument('-s', '--seed', type=int, help="the random seed", default=0)
args = parser.parse_args()

dataset_dir = args.data_dir
if args.dataset_name == 'IMDBB':
    dataset = TUDataset(root=dataset_dir, name="IMDB-BINARY", transform=OneHotDegree(135)) #PyG has a bug here, there should be 2 classes
elif args.dataset_name == "IMDBM":
    dataset = TUDataset(root=dataset_dir, name="IMDB-MULTI", transform=OneHotDegree(88))
elif args.dataset_name == "PROTEINS":
    dataset = TUDataset(root=dataset_dir, name="PROTEINS")
elif args.dataset_name == "MUTAG":
    dataset = TUDataset(root=dataset_dir, name="MUTAG")

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
            'values': [8,12,16]
        },
        'learning_rate': {
            'values': [0.0001, 0.0005, 0.001, 0.0015]
        },
        'smooth_fac': {
            'value': 0.5 #doesn't matter
        },
        'hid_dim': {
            'value': args.hid_dim
        },
        'weight_decay': {
            'values': [0, 1e-5]
        },
        'num_epochs': {
            'value': 200
        },
        'mlp_dropout': {
            'values': [0.4, 0.6]
        },
        'gnn_dropout': {
            'values': [0, 0.5]
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
        'batch_size':{
            'value': args.batch_size
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
        }
    }
    sweep_config['parameters'] = parameters_dict
else:
    run_config = {
        'num_iter_layers': args.layer_num,
        'learning_rate': args.learning_rate,
        'smooth_fac': 0.5,
        'hid_dim':args.hid_dim,
        'weight_decay':args.weight_decay,
        'num_epochs':args.num_epochs,
        'mlp_dropout':args.mlp_dropout,
        'gnn_dropout':args.gnn_dropout,
        'dataset_name':args.dataset_name,
        'layer_name':args.layer_name,
        'readout_name':args.readout_name,
        'homo_flag':args.homo_flag,
        'batch_size': args.batch_size,
        'arc_name':args.arc_name,
        'encoder_layer_num':args.encoder_layer_num,
        'decoder_layer_num':args.decoder_layer_num,
        'pct_start': 0.2,
        'lr_scheduler':'OneCycleLR',
        'jk': args.jk_type
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channel, out_channel = dataset.num_features, dataset.num_classes
    train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)

    encoder_mlp = MLP(
                    in_channels=in_channel,
                    hidden_channels=config.hid_dim,
                    out_channels=config.hid_dim,
                    num_layers=config.encoder_layer_num
                )
    
    layer_num = config.num_iter_layers
    
    if args.gnn_as_encoder: 
        encoder = GINConv(encoder_mlp) # note the GINConv here is the one from PyG, which is not the same as that inside our GNNs
        if args.arc_name == "DeepGNN":
            layer_num = config.num_iter_layers - 1 # for GIN, in this case, the encoder already counts as one layer

    else:
        encoder = encoder_mlp
    
    if args.jk_type == 'cat':
        decoder_in_channel = config.hid_dim * (config.num_iter_layers + 1) #since we are concatenating READOUT results from all iterations/layers
    else:
        decoder_in_channel = config.hid_dim
    decoder = MLP(
                    in_channels=decoder_in_channel,
                    hidden_channels=decoder_in_channel,
                    out_channels=out_channel,
                    num_layers=config.decoder_layer_num
                )

    if args.our_model:
        net = GraphIterativeGNN(layer_name=config.layer_name,
                           hidden_size=config.hid_dim,
                           train_schedule=train_schedule,
                           homogeneous_flag=config.homo_flag,
                           mlp_dropout=config.mlp_dropout,
                           gnn_dropout=config.gnn_dropout,
                           readout_name=config.readout_name,
                           encoder=encoder,
                           decoder=decoder,
                           jk=args.jk_type
                           )
    else:
        net = GraphGNNModels(architecture_name=config.arc_name,
                        layer_num=layer_num,
                        layer_name=config.layer_name,
                        hidden_size=config.hid_dim,
                        readout_name=config.readout_name,
                        homogeneous_flag=config.homo_flag,
                        confidence_layer_num=1,
                        encoder=encoder,
                        decoder=decoder,
                        jk=args.jk_type
                        )
    train_with_cross_val(args.num_folds, dataset, net, args.seed, cross_entropy, config, device)


    
    

if args.mode == 'Sweep':
    sweep_id = wandb.sweep(sweep_config, project="IterGNN")
    wandb.agent(sweep_id, run_exp)
else:
    run_exp(run_config)

