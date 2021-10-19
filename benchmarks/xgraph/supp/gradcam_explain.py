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
from tqdm import tqdm
import pickle as pk
from torch_geometric.utils.loop import add_self_loops, remove_self_loops,add_remaining_self_loops
from dig.xgraph.utils import *
import matplotlib.pyplot as plt
from torch_geometric.datasets import CitationFull, Planetoid
import torch_geometric.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN2', dest='gnn models')
parser.add_argument('--model_name', default='GCN2')
parser.add_argument('--model_level', default='node')
parser.add_argument('--dim_hidden', default=64)
parser.add_argument('--alpha', default=0.5)
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask
def split_dataset(dataset):
    indices = []
    num_classes = 8
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

if parser.dataset_name != 'Cora':
    dataset = get_dataset(parser)
else:
    dataset = Planetoid('./datasets', 'Cora',split="public", transform = T.NormalizeFeatures())
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
# model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=20, num_classes=num_classes)
#
# model.to(device)
# check_checkpoints()
# # ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN2','GCN2_best.pth')
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l','GCN_2l_best.pth')
# model.load_state_dict(torch.load(ckpt_path)['net'])
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
# model.load_state_dict(torch.load(ckpt_path)['state_dict'])
model = GM_GCN(num_layers, dim_node, dim_hidden, num_classes)
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GM_GCN','GM_GCN_100_best.pth')
ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN','GM_GCN_100_nopre_best.pth')
model.load_state_dict(torch.load(ckpt_path)['net'])
model.to(device)
# model = GAT(num_layers, dim_node, 300, num_classes, heads = [7,4,1])
# model.to(device)
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GAT','GAT_100_best.pth')
# model.load_state_dict(torch.load(ckpt_path)['net'])

# node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
# data = dataset[0].cuda()
# node_idx = node_indices[6]


from dig.xgraph.method import GradCAM
explainer = GradCAM(model, explain_graph=False)



# visualize
# sparsity = 0.5
#
# from dig.xgraph.method.pgexplainer import PlotUtils
# plotutils = PlotUtils(dataset_name='ba_community')
# for index, data in enumerate(dataloader):
#     data= dataset.data
#     data.to(device)
#     motif = pk.load(open('Ba_Community_motif.plk','rb'))
#     # node_indices = torch.where(data.test_mask * (data.y.float()%4 !=0))[0].tolist()
#     node_indices = list(motif.keys())
#     for node_idx in tqdm(node_indices, total=len(node_indices)):
#         node_idx = 328
#         hard_mask, edge_masks, subset= \
#             explainer(data, control_sparsity = False, sparsity=sparsity, num_classes=num_classes, node_idx=node_idx,y = data.y)
#         # print(edge_masks[data.y[node_idx]].shape, add_self_loops(data.edge_index)[0].shape)
#         # sys.exit()
#         edge_mask = edge_masks[data.y[node_idx]]
#         # self_loop = add_self_loops(data.edge_index)[0]
#         new_edge_index = data.edge_index[:, hard_mask]
#         # edge_mask = edge_mask[hard_mask]
#         # real_gate = []
#         # for i in range(new_edge_index.shape[-1]):
#         #     temp = new_edge_index[:, i]
#         #     if temp[0] != temp[1]:
#         #         real_gate.append(i)
#         # new_edge_index = remove_self_loops(new_edge_index)[0]
#         # edge_mask = edge_mask[real_gate]
#         new_data = Data(x = data.x[subset], edge_index= new_edge_index, y= data.y[subset])
#         visualize(new_data, edge_mask=edge_mask, top_k=6, plot_utils=plotutils, node_idx= node_idx, vis_name = f'fig/gradcam_bacom_gcn100/{node_idx}.pdf')
#         plt.show()
#         sys.exit()
# sys.exit()











a = 0.5
b = 0.05
c = []
while a < 1:
    c.append(a)
    a += b
# --- Set the Sparsity to 0.5




# large_index = pk.load(open('large_subgraph_bacom.pk','rb'))['node_idx']
# motif = pk.load(open('Ba_Community_motif.plk','rb'))
sparsitys = []
for index, data in enumerate(dataloader):
    # node_indices = list(set(large_index).intersection(set(motif.keys())))
    data = dataset[0]
    # node_indices = list(set(large_index).intersection(set(motif.keys())))
    # node_indices = list(motif.keys())
    node_indices = torch.where(data.test_mask)[0]
    spar = [0 for e in c]
    for j, node_idx in tqdm(enumerate(node_indices), total = len(node_indices)):
        import random
        import numpy as np
        torch.manual_seed(42)
        random.seed(0)
        np.random.seed(0)
        data.to(device)
        _, _, _, masks, related_preds = \
            explainer(data, num_classes=num_classes, node_idx=node_idx,
                       evaluation_confidence = c,control_sparsity = False)
        for i in range(len(spar)):
            spar[i] += related_preds[i]
for i in range(len(spar)):
    sparsity = spar[i]/(j + 1)
    sparsitys.append(sparsity)
    print('GCN:',f'Evaluation Confidence: {c[i]:.2f}\n'
                 f'Sparsity: {sparsity:.4f}\n')
temp = {'evaluation_confidence':c,
        'sparsity':sparsitys}
print(sparsitys)