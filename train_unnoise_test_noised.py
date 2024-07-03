#!/usr/bin/env python
# coding=utf-8

import sys

from main.train import train_with_cross_val, CVeval, CVtrain
from main.Planetoid_utils import add_noise
from myModels.GNNModel import GraphGNNModels
from myModels.ourModels import GraphIterativeGNN
from myModels.utils import make_uniform_schedule
from argparse import ArgumentParser
import torch
import numpy as np
import torch.nn.functional as F
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
parser.add_argument('-np', '--noise_percent', type=float, help="noise percent", default=0)

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
            'values': [4,8,12,16]
        },
        'learning_rate': {
            'values': [0.0001, 0.0005, 0.001, 0.0015]
        },
        'smooth_fac': {
            'values': [0.5, 0.7] #doesn't matter
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
            'value': "MUTAG"
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
        },
        "noise_percent":{
            'value':args.noise_percent
        },
        'our_model':{
            'value':args.our_model
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
        'dataset_name':"MUTAG",
        'layer_name':args.layer_name,
        'readout_name':args.readout_name,
        'homo_flag':args.homo_flag,
        'batch_size': args.batch_size,
        'arc_name':args.arc_name,
        'encoder_layer_num':args.encoder_layer_num,
        'decoder_layer_num':args.decoder_layer_num,
        'pct_start': 0.2,
        'lr_scheduler':'OneCycleLR',
        'jk': args.jk_type,
        'noise_percent':args.noise_percent
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

    encoder = MLP(
                    in_channels=in_channel,
                    hidden_channels=config.hid_dim,
                    out_channels=config.hid_dim,
                    num_layers=config.encoder_layer_num
                )
    
    layer_num = config.num_iter_layers
    decoder = MLP(
                    in_channels=config.hid_dim,
                    hidden_channels=config.hid_dim,
                    out_channels=out_channel,
                    num_layers=config.decoder_layer_num
                )

    if args.our_model:
        model = GraphIterativeGNN(layer_name=config.layer_name,
                           hidden_size=config.hid_dim,
                           train_schedule=train_schedule,
                           homogeneous_flag=config.homo_flag,
                           mlp_dropout=config.mlp_dropout,
                           gnn_dropout=config.gnn_dropout,
                           readout_name=config.readout_name,
                           encoder=encoder,
                           decoder=decoder,
                           jk=args.jk_type,
                           use_bn=False
                           )
    else:
        model = GraphGNNModels(architecture_name=config.arc_name,
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
    
    total_len = len(dataset)
    total_idx = np.arange(total_len)
    np.random.seed(42) 
    np.random.shuffle(total_idx)
    train_idx = total_idx[0:150]
    test_idx = total_idx[150:188]
    train_set = dataset[train_idx]
    test_set = dataset[test_idx]
    for test_graph_idx in range(len(test_set)):
        noise = torch.tensor(np.random.normal(0, config.noise_percent, test_set[test_graph_idx].x.size()))
        test_set[test_graph_idx].x += noise
    print("Gaussian noise with mean 0, std {} is added!".format(config.noise_percent))
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    model.to(device).reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.num_epochs, steps_per_epoch=len(train_loader), pct_start=config.pct_start)
    for epoch in range(config.num_epochs):
        CVtrain(model, train_loader, optimizer, scheduler, F.cross_entropy, device)
        train_loss, train_acc = CVeval(model, train_loader, F.cross_entropy, device)
        test_loss, test_acc = CVeval(model, test_loader, F.cross_entropy, device)
        wandb.log({
            "epoch": epoch,
            "Train accuracy": train_acc,
            "Test loss": test_loss,
            "Test accuracy": test_acc
        })
        print("Epoch {}: train loss: {:.4}, train accuracy: {:.4}, test loss: {:.4}, test accuracy: {:.4}".format(epoch+1, train_loss, train_acc, test_loss, test_acc))
    test_loss, test_acc = CVeval(model, test_loader, F.cross_entropy, device)
    print("EXP ended! TEST LOSS: {:.4}, TEST ACCURACY:{:.4}".format(test_loss, test_acc))

    
    

if args.mode == 'Sweep':
    sweep_id = wandb.sweep(sweep_config, project="IterGNN")
    wandb.agent(sweep_id, run_exp)
else:
    run_exp(run_config)

