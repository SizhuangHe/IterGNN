#!/usr/bin/env python
# coding=utf-8

import sys

from main import train_model
from main.utils import  weighted_cross_entropy
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
import torch.nn.functional as F
import wandb
wandb.login()
# you don't want to change information about the generated dataset in your sweep
parser = ArgumentParser()

parser.add_argument('-dir', '--dataset_root', type=str, help="the path to the dataset", default="/vast/palmer/scratch/dijk/sh2748/data/data/LRGB")
parser.add_argument('-dn', '--dataset_name', type=str, help="the name of the LRGB dataset", choices=["PascalVOC-SP", "Peptides-func", "Peptides-struct"],default="PascalVOC-SP")
parser.add_argument('-m', '--mode', type=str, choices=['Sweep', 'Run'], help="whether to initiate a wandb sweep or a single run", default='Run')
parser.add_argument('-dw', '--disable_wandb', action='store_true')
parser.add_argument('-wr', '--wandb_root', type=str, help="the path to the wandb folder", default="/vast/palmer/scratch/dijk/sh2748")
parser.add_argument('-cpf', '--checkpoint_freq', type=int, help="checkpoint every this number of epochs", default=30)

parser.add_argument('-bs', '--batch_size', type=int, help="the batch size for mini-batch", default=32)

parser.add_argument('-o', '--our_model', action='store_true')
parser.add_argument('-hd', '--hid_dim', type=int, help="the hidden dimension of the model", default=64)
parser.add_argument('-a', '--arc_name', choices=['IterGNN', 'DeepGNN'], help="the architecture of the GNN", default='IterGNN')
parser.add_argument('-ln', '--layer_num', type=int, 
                    help="if using a DeepGNN, this is the number of layers; if using our iterGNN, this is the number of iterations; if using their iterGNN, this is the MAX number of iterations", default=30)
parser.add_argument('-l', '--layer_name', type=str, choices=['GCNConv', 'GATConv','GINConv','SAGEConv'], help="the GNN layer to build up the model", default='GCNConv')
parser.add_argument('-rn', '--readout_name', type=str, choices=['Max', 'Mean'], help="the name of the readout module", default='Max')
parser.add_argument('-hf', '--homo_flag',action='store_true')
parser.add_argument('-eln', '--encoder_layer_num', type=int, help="the number of layers in the encoder, for VOCSP, this is ignored and defaulted to 1", default=2)
parser.add_argument('-dln', '--decoder_layer_num', type=int, help="the number of layers in the decoder", default=1)
parser.add_argument('-mdo', '--mlp_dropout', type=float, help="the dropout rate of the final classification layers", default=0.5)
parser.add_argument('-gdo', '--gnn_dropout', type=float, help="the dropout rate of the GNN layers", default=0.5)
parser.add_argument('-bn', '--use_bn', action='store_true')
parser.add_argument('-rl', '--use_relu', action='store_true')
parser.add_argument('-sf', '--smooth_factor', type=float, help="the smoothing factor", default=0.5)

parser.add_argument('-lr', '--learning_rate', type=float, help="learning rate", default=0.0002)
parser.add_argument('-wd', '--weight_decay', type=float, help="the weight decay", default=1e-5)
parser.add_argument('-ne', '--num_epochs', type=int, help="the number of epochs", default=300)
parser.add_argument('-ps', '--pct_start', type=float, help="pct_start for torch.optim.lr_scheduler.OneCycleLR", default=0.2)

parser.add_argument('-s', '--seed', type=int, help="the random seed to run the experiments with", default=-1)
parser.add_argument('--save_checkpoint', action='store_true')
parser.add_argument('--model_save_path', help="the path to save model dicts to")

args = parser.parse_args()


sweep_config = {
    'method': 'grid'
}

metric = {
    'name': 'Test F1',
    'goal': 'maximize'
}
sweep_config['metric'] = metric

parameters_dict = {
    'num_iter_layers': {
        'value': args.layer_num
    },
    'learning_rate': {
        'values': [0.0001, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.0025]
    },
    'smooth_fac': {
        'value': args.smooth_factor
    },
    'hid_dim': {
        'value': args.hid_dim
    },
    'weight_decay': {
        'value': args.weight_decay
    },
    'num_epochs': {
        'value': args.num_epochs
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
    'use_bn':{
        'value': args.use_bn
    },
    'use_relu':{
        'value': args.use_relu
    },
    'mlp_dropout':{
        'value': args.mlp_dropout
    },
    'gnn_dropout':{
        'value': args.gnn_dropout
    },
    'seed':{
        'value': args.seed
    },
    'dataset_name':{
        'value':args.dataset_name
    },
    "checkpoint_freq":{
        'value': args.checkpoint_freq
    },
    "our_model":{
        'value': args.our_model
    },
    "model_save_path":{
        'value': args.model_save_path
    },
    "save_checkpoint":{
        'value': args.save_checkpoint
    }      
}
sweep_config['parameters'] = parameters_dict


if args.disable_wandb:
    wandb_mode = 'disabled'
else:
    wandb_mode = None


dataset_root = args.dataset_root
dataset_name = args.dataset_name
train_dataset = LRGBDataset(root=dataset_root, name=dataset_name, split="train")
val_dataset = LRGBDataset(root=dataset_root, name=dataset_name, split="val")
test_dataset = LRGBDataset(root=dataset_root, name=dataset_name, split="test")

if dataset_name == "PascalVOC-SP":
    task = "Node"
    criterion = weighted_cross_entropy
elif dataset_name == "Peptides-func":
    task = "Graph"
    criterion = F.cross_entropy

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterGNN", 
               config=config, 
               notes="Sweep for the IterGNN, from McCleary",
               tags=["IterGNN"],
               mode=wandb_mode)
    config = wandb.config
    print(str(config))
    
    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,config.batch_size, shuffle=False)
    in_channel = train_dataset.num_node_features
    edge_channel = train_dataset.num_edge_features
    out_channel = train_dataset.num_classes

    print()
    print("### Dataset statistics ###")
    print("Dataset name: {}".format(config.dataset_name))
    print("Overall dataset size: {}, training set: {}, validation set: {}, test set: {}".format(len(train_dataset)+len(val_dataset)+len(test_dataset),
                                                                                                    len(train_dataset),
                                                                                                    len(val_dataset),
                                                                                                    len(test_dataset)))
    print("Number of node features: {}".format(in_channel))
    print("Number of edge features: {}".format(edge_channel))
    print("Number of classes: {}".format(out_channel))
    print("##########################")
    print()

    train_model(train_loader, val_loader,test_loader, in_channel, edge_channel, out_channel,task,config, our_model = args.our_model, loss_func=criterion)


sweep_id = wandb.sweep(sweep_config, project="IterGNN")
wandb.agent(sweep_id, run_exp)

