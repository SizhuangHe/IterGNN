from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
import torch.nn.functional as F
from argparse import ArgumentParser
import torch
from myModels.ourModels import NodeIterativeGNN
from myModels.OtherLayers import VOCNodeEncoder, GNNInductiveNodeHead
from myModels.utils import make_uniform_schedule
from main.train import eval_lrgb
import random
import numpy as np
from main.utils import  weighted_cross_entropy
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = ArgumentParser()

parser.add_argument('-dir', '--dataset_root', type=str, help="the path to the dataset", default="/vast/palmer/scratch/dijk/sh2748/data/data/LRGB")
parser.add_argument('-dn', '--dataset_name', type=str, help="the name of the LRGB dataset", choices=["PascalVOC-SP", "Peptides-func", "Peptides-struct"],default="PascalVOC-SP")
parser.add_argument('--max_iter', type=int, help="the maximum number of iterations to run the model on", default=40)
parser.add_argument('--smooth_factor', type=float, help="the smoothing factor to use during inference")


args = parser.parse_args()

dataset_root = args.dataset_root
dataset_name = args.dataset_name
val_dataset = LRGBDataset(root=dataset_root, name=dataset_name, split="val")
test_dataset = LRGBDataset(root=dataset_root, name=dataset_name, split="test")
val_loader = DataLoader(val_dataset, 32, shuffle=False)
test_loader = DataLoader(test_dataset, 32, shuffle=False)

seed = 12345
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

val_losses = []
val_f1s = []
test_losses = []
test_f1s = []

encoder = VOCNodeEncoder(emb_dim=400)
decoder = GNNInductiveNodeHead(400, 400, 21, 3)
for i in tqdm(range(1, args.max_iter)):
    train_schedule = make_uniform_schedule(i, args.smooth_factor)
    net = NodeIterativeGNN(layer_name="GCNConv",
                            hidden_size=400,
                            train_schedule=train_schedule,
                            homogeneous_flag=False,
                            gnn_dropout=0,
                            mlp_dropout=0,
                            readout_name="Max",
                            encoder=encoder,
                            decoder=decoder,
                            use_bn=False,
                            use_relu=True,
                            ).to(device)
    net.load_state_dict(torch.load("model_dicts/1-iter_model/OUR+IterGNN+PascalVOC-SP+-1epoch+final.pth")["model"])
    test_loss, test_f1 = eval_lrgb(net, test_loader, weighted_cross_entropy, device, 'PascalVOC-SP')
    val_loss, val_f1 = eval_lrgb(net, val_loader, weighted_cross_entropy, device, 'PascalVOC-SP')
    val_losses.append(val_loss)
    val_f1s.append(val_f1)
    test_losses.append(test_loss)
    test_f1s.append(test_f1)
np.save('plot_data/val_loss.npy', val_losses)
np.save('plot_data/val_f1.npy', val_f1s)
np.save('plot_data/test_loss.npy', test_losses)
np.save('plot_data/test_f1.npy', test_f1s)
val_losses = np.load('plot_data/val_loss.npy')
val_f1s = np.load('plot_data/val_f1.npy')
test_losses = np.load('plot_data/test_loss.npy')
test_f1s = np.load('plot_data/test_f1.npy')

fig, axs = plt.subplots(2,2)
axs[0,0].plot(val_losses)
axs[0,0].set_title("Validation loss")
axs[0, 0].set_yscale('log')
axs[0,1].plot(val_f1s)
axs[0,1].set_title("Validation F1")
axs[0, 1].set_yscale('log')
axs[1,0].plot(test_losses)
axs[1,0].set_title("Test loss")
axs[1, 0].set_yscale('log')
axs[1,1].plot(test_f1s)
axs[1,1].set_title("Test F1")
axs[1, 1].set_yscale('log')
for ax in axs.flat:
    ax.set(xlabel='number of iterations')
for ax in axs.flat:
    ax.label_outer()

plt.savefig("plot.png")