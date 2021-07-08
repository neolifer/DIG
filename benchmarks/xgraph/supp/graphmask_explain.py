import sys
sys.path.append('../..')

sys.path.append('../../..')


# from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.models import *
from dig.xgraph.utils import *
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
import os.path as osp
import os
import argparse
from PGExplainer.load_dataset import get_dataset, get_dataloader
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN2', dest='gnn models')
parser.add_argument('--model_name', default='GCN2')
parser.add_argument('--model_level', default='node')
parser.add_argument('--dim_hidden', default=300)
parser.add_argument('--alpha', default=0.5)
parser.add_argument('--theta', default=0.5)
parser.add_argument('--num_layers', default=8)
parser.add_argument('--shared_weights', default=False)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--dataset_dir', default='./datasets/')
parser.add_argument('--dataset_name', default='BA_community')
parser.add_argument('--epoch', default=1000)
parser.add_argument('--save_epoch', default=10)
parser.add_argument('--lr', default=0.01)
parser.add_argument('--wd1', default=1e-3)
parser.add_argument('--wd2', default=1e-5)
parser = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split_dataset(dataset):
    indices = []
    num_classes = 4
    train_percent = 0.7
    for i in range(num_classes):
        index = (dataset.data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)

    rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    dataset.data.train_mask = index_to_mask(train_index, size=dataset.data.num_nodes)
    dataset.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=dataset.data.num_nodes)
    dataset.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=dataset.data.num_nodes)

    dataset.data, dataset.slices = dataset.collate([dataset.data])

    return dataset

dataset = get_dataset(parser)
# dim_node = dataset.num_node_features
dataset.data.x = dataset.data.x.to(torch.float32)

# dataset.data.x = dataset.data.x[:, :1]
# dataset.data.y = dataset.data.y[:, 2]
dim_node = dataset.num_node_features

dim_edge = dataset.num_edge_features
# num_targets = dataset.num_classes
num_classes = dataset.num_classes

splitted_dataset = split_dataset(dataset)
splitted_dataset.data.mask = splitted_dataset.data.test_mask
splitted_dataset.slices['mask'] = splitted_dataset.slices['train_mask']
dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)


def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)
model_level = parser.model_level

dim_hidden = parser.dim_hidden

alpha = parser.alpha
theta=parser.theta
num_layers=parser.num_layers
shared_weights=parser.shared_weights
dropout=parser.dropout

# model = GCN2_mask(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
#                    shared_weights, dropout)
model = GM_GCN(n_layers = num_layers, input_dim = dim_node, hid_dim = dim_hidden, n_classes = num_classes)

model.to(device)
check_checkpoints()
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN2','GCN2_best.pth')
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l','GCN_2l_best.pth')
ckpt_path = osp.join('checkpoints', 'ba_community', 'GM_GCN','GM_GCN_best.pth')
model.load_state_dict(torch.load(ckpt_path)['net'])
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
# model.load_state_dict(torch.load(ckpt_path)['state_dict'])

from dig.xgraph.method import GraphMaskExplainer, GraphMaskAdjMatProbe
message_dims = [dim_hidden for _ in range(num_layers)]
hidden_dims = [dim_hidden for _ in range(num_layers)]
vertex_dims = [3*dim_hidden]+[3*dim_hidden for _ in range(num_layers - 1)]
GraphMask = GraphMaskAdjMatProbe(vertex_dims, message_dims, num_classes, hidden_dims )
model.cuda()
GraphMask.cuda()

explainer = GraphMaskExplainer(model, GraphMask)

explainer.train_graphmask(splitted_dataset)
torch.save(GraphMask.state_dict(), 'tmp.pt')
state_dict = torch.load('tmp.pt')
GraphMask.load_state_dict(state_dict)


from dig.xgraph.method.pgexplainer import PlotUtils
plotutils = PlotUtils(dataset_name='ba_community')

node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
from dig.xgraph.method.pgexplainer import PlotUtils
plotutils = PlotUtils(dataset_name='ba_community')
data = dataset[0].cuda()
node_idx = node_indices[6]
new_data, subset, new_node_idx, mask= \
    explainer(data, node_idx=node_idx, y=data.y, top_k=6, visualize = True)

visualize(new_data, edge_mask=mask, top_k=6, plot_utils=plotutils, node_idx= new_node_idx, vis_name = 'graphmask.png')
# sys.exit()

# --- Create data collector and explanation processor ---
from dig.xgraph.evaluation import XCollector
x_collector = XCollector()

## Run explainer on the given model and dataset
index = -1
for i, data in enumerate(dataloader):
    for j, node_idx in enumerate(torch.where(data.test_mask)[0].tolist()):
        index += 1
        print(f'explain graph {i} node {node_idx}')
        data.to(device)

        if torch.isnan(data.y[0].squeeze()):
            continue

        walks, masks, related_preds = \
            explainer(data, node_idx=node_idx, y=data.y, top_k=6)

        x_collector.collect_data(masks, related_preds)

        # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
        # obtain the result: x_processor(data, masks, x_collector)
        if index >= 99:
            break

    if index >= 99:
        break

print(f'Fidelity: {x_collector.fidelity:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')
