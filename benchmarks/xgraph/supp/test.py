import sys
sys.path.append('../..')
sys.path.append('../../..')
from torch_geometric.datasets import CitationFull, Planetoid
import torch_geometric.transforms as T
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt
from dig.xgraph.models import *
import os.path as osp
import torch
import argparse

tsne = TSNE(n_components=2, random_state=0)
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN2', dest='gnn models')
parser.add_argument('--model_name', default='GCN2')
parser.add_argument('--model_level', default='node')
parser.add_argument('--dim_hidden', default=64)
parser.add_argument('--alpha', default=0.1)
parser.add_argument('--theta', default=0.5)
parser.add_argument('--num_layers', default=2)
parser.add_argument('--shared_weights', default=False)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--dataset_dir', default='./datasets/')
parser.add_argument('--dataset_name', default='Cora')
parser.add_argument('--epoch', default=1000)
parser.add_argument('--save_epoch', default=10)
parser.add_argument('--lr', default=0.01)
parser.add_argument('--wd1', default=1e-3)
parser.add_argument('--wd2', default=1e-5)
parser = parser.parse_args()
dataset = Planetoid('./datasets', 'Cora',split="public", transform = T.NormalizeFeatures())
dataset.data.x = dataset.data.x.to(torch.float32)

dim_node = dataset.num_node_features

dim_edge = dataset.num_edge_features
model_level = parser.model_level

dim_hidden = parser.dim_hidden

alpha = parser.alpha
theta=parser.theta
num_layers=parser.num_layers
shared_weights=parser.shared_weights
dropout=parser.dropout
num_classes = dataset.num_classes
model = GM_GCN(n_layers = num_layers, input_dim = dim_node, hid_dim = dim_hidden, n_classes = num_classes)

model.to('cuda:0')
ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN','GM_GCN_100_nopre_best.pth')

model.load_state_dict(torch.load(ckpt_path)['net'])
model.set_get_vertex(False)

data = dataset.data
data.cuda()

vertex = model.get_emb(data.x, data.edge_index)
result = tsne.fit_transform(vertex.cpu().detach().numpy())

x = result[:, 0]
y = result[:,1]
sns.scatterplot(x = x , y = y, hue = dataset.data.y.cpu().numpy())
plt.show()