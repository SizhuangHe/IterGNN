#!/usr/bin/env python
# coding=utf-8

import torch, os, sys, logging, time, copy
from .parameters import params2path, param2dataset, param2model, get_dataset_param
import torch.nn.functional as F

def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        log_file = os.path.join(save_dir, 'log.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_channels(dataset):
    if dataset.__class__.__name__ == 'DatasetParam': #for debug
        in_channel, edge_channel, out_channel = 3, 1, 1
    else: #is then the dataset
        in_channel = dataset.num_node_features
        edge_channel = dataset.num_edge_features
        out_channel = dataset.num_classes
    return in_channel, edge_channel, out_channel

def make_data(dataset_param, test_min_num_node):
    dataset_param.device = torch.device('cpu')

    # dataset
    dataset_size = dataset_param.size

    start_time = time.time()
    train_dataset = param2dataset(dataset_param, train_flag=True)
    print('train_dataset generated: %.2f seconds'%(time.time()-start_time))

    start_time = time.time()
    test_dataset_param = copy.deepcopy(dataset_param)
    test_dataset_param.size = 1000
    test_dataset_param.min_num_node = test_min_num_node
    test_dataset = param2dataset(test_dataset_param, train_flag=False)
    print('test_dataset generated: %.2f seconds'%(time.time()-start_time))

    return train_dataset, test_dataset

def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    
    # calculating label weights for weighted loss computation
    V = true.size(0)
    
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()

    # multiclass
    if pred.ndim > 1:
        pred = F.log_softmax(pred, dim=-1)
        loss = F.nll_loss(pred, true, weight=weight)
        
        return loss
    # binary
    else:
        loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                    weight=weight[true])
        return loss