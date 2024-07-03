import myModels.GNNAgg as agg
import myModels.GNNArch as arc
import myModels.GNNLayers as layers
import myModels.GNNModel as models
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn.models import MLP
from torch_geometric.profile.utils import count_parameters
import numpy as np
import torch
import math
import torch.nn.functional as F
from myModels.GNNModel import NodeGNNModels
from myModels.ourModels import NodeIterativeGNN
from myModels.utils import make_uniform_schedule
import wandb
wandb.login()


def add_noise(data, percent=0, seed=None):
    #add random 1's to data
    if percent > 0 and percent <= 1:
        len = np.prod(list(data.x.shape))
        ones = math.floor(len * percent)
        zeros = len - ones
        noise = torch.cat((torch.zeros(zeros), torch.ones(ones)))
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
            noise = noise[torch.randperm(noise.size(0), generator=rng)]
        else:
            noise = noise[torch.randperm(noise.size(0))]
        noise = torch.reshape(noise, data.x.shape)
        data.x += noise
    return data
def accuracy(guess, truth):
    correct = guess == truth
    acc = correct.sum().item() / truth.size(dim=0)
    return acc

def train_epoch(model, data, optimizer, scheduler, device):
    model.train()
    data = data.to(device)
    
    output, num_iter = model(data)
    loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
    pred = F.log_softmax(output[data.train_mask], dim=1).argmax(dim=1)
    acc = accuracy(pred, data.y[data.train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()
    wandb.log({
            "num_iter": num_iter
        })
    
    return loss, acc

def validate_epoch(model, data, device):
    model.eval()
    data = data.to(device)

    output, _ = model(data)
    loss = F.cross_entropy(output[data.val_mask], data.y[data.val_mask])
    pred = F.log_softmax(output[data.val_mask], dim=1).argmax(dim=1)
    acc = accuracy(pred, data.y[data.val_mask])
    return loss, acc

def train(model, data, optimizer, scheduler, config, device):
    
    for epoch in range(config.num_epochs):
        loss_train, acc_train = train_epoch(model, data, optimizer, scheduler, device)
        loss_val, acc_val = validate_epoch(model, data, device)
        wandb.log({
            'training_loss': loss_train,
            'training_accuracy': acc_train,
            'validation_loss': loss_val,
            'validation_accuracy': acc_val,
            "epoch": epoch+1,
        })
        print("Epoch {}, train loss: {:.4}, train accuracy: {:.4}, validation loss: {:.4}, validation accuracy: {:.4}".format(epoch+1, loss_train, acc_train, loss_val, acc_val))

        
  
def test(model, data, device):
    model.eval()
    data = data.to(device)

    output, _ = model(data)
    loss = F.cross_entropy(output[data.test_mask], data.y[data.test_mask])
    pred = F.log_softmax(output[data.test_mask], dim=1).argmax(dim=1)
    acc = accuracy(pred, data.y[data.test_mask])
    
    return loss, acc

def exp_per_model(model, data, optimizer, scheduler, config, device):
    num_params = count_parameters(model)
    wandb.log({ 
            'num_param': num_params
    }) 
    train(model, data, optimizer, scheduler, config, device)
    loss_test, acc_test = test(model, data, device)
    wandb.log({
        'test_loss': loss_test,
        'test_accuracy': acc_test
    })
    print("Test loss: {:.4}, Test accuracy: {:.4}".format(loss_test, acc_test))
