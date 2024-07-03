#!/usr/bin/env python
# coding=utf-8

from .parameters import params2path, param2dataset, param2model, get_dataset_param
from .utils import getDevice, setup_logger, weighted_cross_entropy
from .evaluate import evaluate
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import numpy as np
import random
from tqdm import tqdm
import torch, os, time, shutil, copy
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch_geometric.loader import DataLoader, DataListLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.profile.utils import count_parameters
from myModels.GNNModel import GraphGNNModels, NodeGNNModels
from myModels.ourModels import GraphIterativeGNN, NodeIterativeGNN
from myModels.utils import make_uniform_schedule
from myModels.OtherLayers import VOCNodeEncoder, GNNInductiveNodeHead, LobsterEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import wandb
wandb.login()

def checkpoint(model, optimizer, filename):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
        }, filename)
    
def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def model_filename(prefix, config, suffix):
    path = "model_dicts/" + prefix
    if config.our_model:
        prefix = "OUR" + "+"
    else:
        prefix = ""
    model_type = config.arc_name
    dataset_name = config.dataset_name
    seed = config.seed
    suffix = str(suffix)
    
    return path+prefix+model_type+"+"+dataset_name+"+"+str(seed)+"epoch+"+suffix+".pth"

def eval_lrgb(model, loader, criterion = weighted_cross_entropy, device='cpu', dataset='PascalVOC-SP'):
    model.eval()
    y_true = []
    y_pred = []
    pred_vals = [] 

    val_loss = 0
    for batched_data in loader: 
        batched_data = batched_data.to(device)
        pred, _ = model(batched_data) # size of pred is [number of nodes, number of features]
        true = batched_data.y
        loss = criterion(pred, true)
        val_loss += loss.item()

        y_true.append(true.detach())
        y_pred.append(pred.detach())

        pred_val = pred.max(dim=1)[1] # pred_val contains actually class predictions
        pred_vals.append(pred_val.detach())
        

    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim = 0).cpu().numpy()
    pred_vals = torch.cat(pred_vals, dim = 0).cpu().numpy()
    if dataset == 'PascalVOC-SP':
        val_metric = f1_score(y_true, pred_vals, average="macro")
    elif dataset == 'Peptides-func':
        val_metric = eval_ap(y_true, y_pred)
    else:
        raise NotImplementedError
        
    return val_loss, val_metric
    

def eval_ap(y_true, y_pred):
        '''
            compute Average Precision (AP) averaged across tasks
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # ignore nan values
                is_labeled = y_true[:,i] == y_true[:,i]
                ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return sum(ap_list)/len(ap_list)

# def running_test(net, test_data_loader, device,
#                  running_metric_name_list, test_loss_hist,
#                  epoch, step, global_step):
#     test_metric_list, _ = evaluate(net, test_data_loader, device,
#                                 parallel_flag=False,
#                                 metric_name_list=running_metric_name_list)
#     for mn in running_metric_name_list:
#         test_loss_hist[mn].append(np.mean(test_metric_list[mn]))
    
#     print("Relative loss: {:.4}".format(test_loss_hist["relative_loss"][-1]) )
#     print("MSE loss: {:.4}".format(test_loss_hist["mse_loss"][-1]))
#     print("Accuracy: {:.4}".format(test_loss_hist["accuracy"][-1]))
    
#     wandb.log({
#         'epoch': epoch,
#         'step in epoch (batch)': step,
#         'global step': global_step,
#         'relative loss': test_loss_hist["relative_loss"][-1],
#         'mse loss': test_loss_hist["mse_loss"][-1],
#         'accuracy': test_loss_hist["accuracy"][-1]
#     })

def train_model(data_loader, val_loader, test_data_loader, in_channel, edge_channel, out_channel, task,config, our_model, loss_func=F.mse_loss):

    print("------------ Experiment begins ------------")

    if config.seed >0:
        # pass in -1 as seed during sweep
        seed = config.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        print("+++ Random seed: {} +++".format(seed))
        print()

    
    lr = config.learning_rate
    epoch_num = config.num_epochs

    device = getDevice()

    if config.dataset_name == "PascalVOC-SP":
        print("Dataset used: {}".format(config.dataset_name))
        encoder = VOCNodeEncoder(emb_dim=config.hid_dim)
        decoder = GNNInductiveNodeHead(config.hid_dim, config.hid_dim, out_channel, config.decoder_layer_num)
    elif config.dataset_name == 'Peptides-func':
        print("Dataset used: {}".format(config.dataset_name))
        encoder = AtomEncoder(config.hid_dim)
        decoder = GNNInductiveNodeHead(config.hid_dim, config.hid_dim, out_channel, config.decoder_layer_num)
    else:
        raise NotImplementedError

    # build network
    if not our_model:
        if task == 'Graph':
            GNNmodel = GraphGNNModels
        elif task == 'Node':
            GNNmodel = NodeGNNModels
        else:
            raise Exception("Invalid task type")
        net = GNNmodel(architecture_name=config.arc_name,
                        layer_num=config.num_iter_layers,
                        layer_name=config.layer_name,
                        hidden_size=config.hid_dim,
                        readout_name=config.readout_name,
                        homogeneous_flag=config.homo_flag,
                        confidence_layer_num=1,
                        encoder=encoder,
                        decoder=decoder,
                        use_relu=config.use_relu,
                        use_bn=config.use_bn)
    else:
        if task == 'Graph':
            GNNmodel = GraphIterativeGNN
        elif task == 'Node':
            GNNmodel = NodeIterativeGNN
        else:
            raise Exception("Invalid task type")
        train_schedule = make_uniform_schedule(config.num_iter_layers, config.smooth_fac)
        net = GNNmodel(layer_name=config.layer_name,
                           hidden_size=config.hid_dim,
                           train_schedule=train_schedule,
                           homogeneous_flag=config.homo_flag,
                           gnn_dropout=config.gnn_dropout,
                           mlp_dropout=config.mlp_dropout,
                           readout_name=config.readout_name,
                           encoder=encoder,
                           decoder=decoder,
                           use_bn=config.use_bn,
                           use_relu=config.use_relu,
                           )
    net = net.to(device)
    print("Model INFO: ", str(net))
    print()
    wandb.log({
        "num_params": count_parameters(net)
    })

    # optimizer related
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5, weight_decay=config.weight_decay)
    scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.num_epochs, steps_per_epoch=len(data_loader), pct_start=config.pct_start)

    # Training loop
    for epoch in range(1, epoch_num+1):
        net.train()
        train_loss = 0
        # training stage
        for step,data in enumerate(data_loader):
            data = data.to(device)
            pred, layer_num = net(data)
            y = data.y
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            wandb.log({
                'train loss': loss.item(),
                'iter_num': layer_num,
                'lr':optimizer.param_groups[0]['lr']
            })
            train_loss += loss.item()
        # validation stage
        val_loss, val_metric = eval_lrgb(net, val_loader, criterion=loss_func, device=device, dataset=config.dataset_name)
        if config.dataset_name == "PascalVOC-SP":
            wandb.log({
                "Validate f1": val_metric,
                "Validate loss": val_loss,
            })
            print("Epoch {:03d}: Train Loss: {:.4}, Validation Loss: {:.4}, Validation F1: {:.6}".format(epoch, train_loss, val_loss, val_metric))
        elif config.dataset_name == "Peptides-func": 
            wandb.log({
                "Validate AP": val_metric,
                "Validate loss": val_loss,
            })
            print("Epoch {:03d}: Train Loss: {:.4}, Validation Loss: {:.4}, Validation AP: {:.6}".format(epoch, train_loss, val_loss, val_metric))
        else:
            raise NotImplementedError
        if (epoch % config.checkpoint_freq == 0) and (config.save_checkpoint):
            model_path = model_filename(config, epoch)
            checkpoint(net, optimizer, model_path)
    # Test stage
    test_loss, test_metric = eval_lrgb(net, test_data_loader, criterion=loss_func, device=device, dataset=config.dataset_name)
    if config.dataset_name == "PascalVOC-SP":
        wandb.log({
            "Test loss": test_loss,
            "Test f1": test_metric
        })
        print("*** Test Loss: {:.4}, Test F1: {:.6} ***".format(test_loss, test_metric))
    elif config.dataset_name == "Peptides-func": 
        wandb.log({
            "Test loss": test_loss,
            "Test AP": test_metric
        })
        print("*** Test Loss: {:.4}, Test AP: {:.6} ***".format(test_loss, test_metric))
    

    
    
    
    model_path = model_filename(config.model_save_path, config, "final")
    checkpoint(net, optimizer, model_path)
    
    
    print("------------ Experiment finishes ------------")

def accuracy(guess, truth):
    '''
    Receives two np arrays
    '''
    correct = guess == truth
    acc = sum(correct) / len(correct)
    return acc

def CVtrain(model, loader, optimizer, scheduler, loss_func,device):
    model.train()
    for batched_data in loader:
        batched_data = batched_data.to(device)
        pred, layer_num = model(batched_data)
        y = batched_data.y
        loss = loss_func(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        wandb.log({
            'train loss': loss.item(),
            'iter_num': layer_num,
            'lr':optimizer.param_groups[0]['lr']
        })
        
def CVeval(model, loader, loss_func, device):
    model.eval()
    eval_loss = 0
    pred_vals = []
    truths = []
    for batched_data in loader:
        batched_data = batched_data.to(device)
        pred, layer_num = model(batched_data)
        y = batched_data.y
        loss = loss_func(pred, y)
        eval_loss += loss.item()
        pred_val = pred.argmax(dim=1)
        # print(y)
        # print(y.detach().cpu().numpy())
        truths = np.concatenate((truths, y.detach().cpu().numpy()))
        pred_vals = np.concatenate((pred_vals, pred_val.detach().cpu().numpy()))
    
    eval_acc = accuracy(pred_vals, truths)
    return eval_loss, eval_acc
        
def train_with_cross_val(num_folds, dataset, model, seed, loss_func,config, device):
    skf = StratifiedKFold(num_folds, shuffle=True, random_state=seed)
    train_loss_lists, train_accuracy_lists, test_loss_lists, test_accuracy_lists = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(torch.zeros(len(dataset)), dataset.data.y)):
        train_set = dataset[train_idx]
        test_set = dataset[test_idx]
        train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=False)
        model.to(device).reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, epochs=config.num_epochs, steps_per_epoch=len(train_loader), pct_start=config.pct_start)
        train_loss_fold, train_accuracy_fold, test_loss_fold, test_accuracy_fold = [], [], [], []
        for epoch in range(config.num_epochs):
            CVtrain(model, train_loader, optimizer, scheduler, loss_func, device)
            train_loss, train_acc = CVeval(model, train_loader, loss_func, device)
            test_loss, test_acc = CVeval(model, test_loader, loss_func, device)
            wandb.log({
                "epoch": epoch,
                "Train accuracy": train_acc,
                "Test loss": test_loss,
                "Test accuracy": test_acc
            })
            train_loss_fold.append(train_loss)
            train_accuracy_fold.append(train_acc)
            test_loss_fold.append(test_loss)
            test_accuracy_fold.append(test_acc)
            wandb.log({
                'fold{} train losses'.format(fold + 1): train_loss_fold,
                'fold{} train accuracies'.format(fold + 1): train_accuracy_fold,
                'fold{} test losses'.format(fold + 1): test_loss_fold,
                'fold{} test accuracies'.format(fold + 1): test_accuracy_fold
            })
            print("Fold {} epoch {}: train loss: {:.4}, train accuracy: {:.4}, test loss: {:.4}, test accuracy: {:.4}".format(fold+1, epoch+1, train_loss, train_acc, test_loss, test_acc) )

        train_loss_lists.append(train_loss_fold)
        train_accuracy_lists.append(train_accuracy_fold)
        test_loss_lists.append(test_loss_fold)
        test_accuracy_lists.append(test_accuracy_fold)
        avg_train_loss_list = np.mean(train_loss_lists, axis=0)
        avg_train_accuracy_list = np.mean(train_accuracy_lists, axis=0)
        avg_test_loss_list = np.mean(test_loss_lists, axis=0)
        avg_test_accuracy_list = np.mean(test_accuracy_lists, axis=0)
        wandb.log({
            'average train losses': avg_train_loss_list,
            'average train accuracies': avg_train_accuracy_list,
            'average test losses': avg_test_loss_list,
            'average test accuracies': avg_test_accuracy_list
        })

    best_epoch = np.argmax(avg_test_accuracy_list)
    best_epoch_accuracy = np.array(test_accuracy_lists)[:, best_epoch]
    mean_test_accuracy = np.mean(best_epoch_accuracy)
    std_test_accuracy = np.std(best_epoch_accuracy)
    wandb.log({
        "Mean Final Test Acc": mean_test_accuracy,
        "STD Final Test Acc": std_test_accuracy
    })
    print("Best epoch is epoch {}".format(best_epoch))
    print("Mean Final Test Acc ", mean_test_accuracy)
    print("STD Final Test Acc ", std_test_accuracy)
        


#======================Testing training=========================================

from .parameters import get_dataset_param, get_model_param, GeneralParam

def test_train_model():
    dataset_param = get_dataset_param(size=1000)
    model_param = get_model_param()
    general_param = GeneralParam()
    # general_param.resume_flag = True

    train_model(dataset_param, model_param, general_param)

if __name__ == '__main__':
    test_train_model()
