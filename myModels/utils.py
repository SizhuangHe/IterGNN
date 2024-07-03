import torch
from torch.nn import LeakyReLU
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as gnn_softmax
import numpy as np

def homo_gnn_softmax(x, index, size=None):
    '''
    re-scale so that homo_softmax(s*x) = homo_softmax(x) when s > 0
    '''
    assert(not torch.sum(torch.isnan(x)))
    x_max = global_max_pool(x, index, size=size)
    assert(not torch.sum(torch.isnan(x_max)))
    x_min = -global_max_pool(-x, index, size=size)
    assert(not torch.sum(torch.isnan(x_min)))
    x_diff = (x_max-x_min)[index]
    assert(not torch.sum(torch.isnan(x_diff)))
    zero_mask = (x_diff == 0).type(torch.float)
    x_diff = torch.ones_like(x_diff)*zero_mask + x_diff*(1.-zero_mask)
    x = x/x_diff
    assert(not torch.sum(torch.isnan(x)))
    return gnn_softmax(x, index, size)
def make_uniform_schedule(length, smooth_fac):
    return np.full(length, smooth_fac)