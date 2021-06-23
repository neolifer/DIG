import sys
sys.path.append('../../..')

sys.path.append('../../../..')
from dig.xgraph.dataset import MoleculeDataset
from dig.xgraph.models import *
import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.data.dataset import files_exist
import os.path as osp
import os
from PGExplainer.load_dataset import get_dataset, get_dataloader
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN2', dest='gnn models')
parser.add_argument('--model_name', default='GCN2')
parser.add_argument('--model_level', default='node')
parser.add_argument('--dim_hidden', default=20)
parser.add_argument('--alpha', default=0.5)
parser.add_argument('--theta', default=0.5)
parser.add_argument('--num_layers', default=5)
parser.add_argument('--shared_weights', default=False)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--dataset_dir', default='./datasets/')
parser.add_argument('--dataset_name', default='BA_shapes')
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
dataset.data.x = dataset.data.x[:, :1]
# dataset.data.y = dataset.data.y[:, 2]
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
# num_targets = dataset.num_classes
num_classes = dataset.num_classes

splitted_dataset = split_dataset(dataset)
splitted_dataset.data.mask = splitted_dataset.data.test_mask
splitted_dataset.slices['mask'] = splitted_dataset.slices['train_mask']
dataloader = DataLoader(splitted_dataset, batch_size=1, shuffle=False)

# from dig.xgraph.models import GIN_3l

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
model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=20, num_classes=num_classes)

model.to(device)
check_checkpoints()
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN2','GCN2_best.pth')
ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l','GCN_2l_best.pth')
model.load_state_dict(torch.load(ckpt_path)['net'])
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
# model.load_state_dict(torch.load(ckpt_path)['state_dict'])


node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
data = dataset[0].cuda()
node_idx = node_indices[6]
from dig.xgraph.method import GradCAM
explainer = GradCAM(model, explain_graph=False)

# --- Set the Sparsity to 0.5 ---
sparsity = 0.5
data = dataset[0].cuda()
# --- Create data collector and explanation processor ---
from dig.xgraph.evaluation import XCollector, ExplanationProcessor
x_collector = XCollector(sparsity)
# x_processor = ExplanationProcessor(model=model, device=device)

for index, data in enumerate(dataloader):
    for j, node_idx in enumerate(torch.where(data.test_mask)[0].tolist()):
        # print(f'explain graph line {dataloader.dataset.indices[index] + 2}')
        data.to(device)

        if torch.isnan(data.y[0].squeeze()):
            continue

        walks, masks, related_preds = \
            explainer(data, sparsity=sparsity, num_classes=num_classes, node_idx=node_idx)

        x_collector.collect_data(masks, related_preds, data.y[0].squeeze().long().item())

        # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
        # obtain the result: x_processor(data, masks, x_collector)
        if index >= 99:
            break
    if index >= 99:
        break

print(f'Fidelity: {x_collector.fidelity:.4f}\n'
      f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')
