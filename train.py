#!/usr/bin/env python
# coding=utf-8

import sys

from mutils import arg2dataset_param, arg2model_param
from main import get_general_param
from main import train_model
from main.utils import make_data, weighted_cross_entropy
from argparse import ArgumentParser
from torch_geometric.loader import DataLoader
import wandb
wandb.login()

# you don't want to change information about the generated dataset in your sweep
parser = ArgumentParser()
parser.add_argument("-d", "--dataset_name", type=str, choices=['lobster', 'knn', 'mesh', 'random'],help="the generated dataset, choose from lobster, knn, mesh, random", default="lobster")
parser.add_argument("-w", "--weighted_flag", type=str, choices=['weighted', 'unweighted'], help="if the graph is weighted or not", default="unweighted")
parser.add_argument('-n', '--min_num_node', type=int, help="the min number of nodes in the generated graph", default=30)
parser.add_argument('tn', '--test_min_num_node', type=int, help="the min number of nodes in the generated TEST graph", default=1000)
parser.add_argument('-bs', '--batch_size', type=int, help="the batch size for mini-batch", default=32)

parser.add_argument('-o', '--our_model', action='store_true')
parser.add_argument('-hd', '--hid_dim', type=int, help="the hidden dimension of the model", default=64)
parser.add_argument('-a', '--arc_name', choices=['IterGNN', 'DeepGNN'], help="the architecture of the GNN", default='IterGNN')
parser.add_argument('-l', '--layer_name', type=str, choices=['GCNConv', 'GATConv'], help="the GNN layer to build up the model", default='GCNConv')
parser.add_argument('-r', '--readout_name', type=str, choices=['Max', 'Mean'], help="the name of the readout module", default='Max')
parser.add_argument('-hf', '--homo_flag',action='store_true')
args = parser.parse_args()

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
        'value': 30
    },
    'learning_rate': {
        'value': 0.001
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
    'dropout': {
        'value': 0.0
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
    'test_min_num_nod':{
        'value': args.test_min_num_node
    }
}
sweep_config['parameters'] = parameters_dict

def run_exp(config=None):
    wandb.init(job_type="Sweep", 
               project="IterGNN", 
               config=config, 
               notes="Sweep for the IterGNN, from McCleary",
               tags=["IterGNN"],
               dir="/vast/palmer/scratch/dijk/sh2748")
    config = wandb.config
    dataset_param = arg2dataset_param(index_generator=args.dataset_name, weighted_flag=args.weighted_flag, min_num_node=args.min_num_node)

    dataset_param.size = 10000
    dataset_param.min_num_node = 4
    dataset_param.num_num_node = 30

    train_dataset, test_dataset = make_data(dataset_param, config.test_min_num_node)
    data_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset,args.batch_size)
    in_channel = train_dataset.num_node_features
    edge_channel = train_dataset.num_edge_features
    out_channel = train_dataset.num_classes


    train_model(data_loader, test_data_loader, in_channel, edge_channel, out_channel,config, our_model = args.our_model)


sweep_id = wandb.sweep(sweep_config, project="IterGNN")
wandb.agent(sweep_id, run_exp)


