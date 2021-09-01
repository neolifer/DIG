import sys
sys.path.append('../../..')

sys.path.append('../../../..')


from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.models import *
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
import os.path as osp
import os
import argparse
from PGExplainer.load_dataset import get_dataset, get_dataloader
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from itertools import product
from tqdm import tqdm
from dig.xgraph.method import PGExplainer
from dig.xgraph.evaluation import XCollector
import random
import numpy as np
import pickle as pk
from torch.nn import functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import CitationFull, Planetoid
import torch_geometric.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN2', dest='gnn models')
parser.add_argument('--model_name', default='GCN')
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
parser.add_argument('--batch', default=1)
parser = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def split_dataset(dataset, num_classes):
    indices = []
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
# print(dataset.data)
# sys.exit()
splitted_dataset = split_dataset(dataset, num_classes)
splitted_dataset.data.mask = splitted_dataset.data.test_mask
splitted_dataset.slices['mask'] = splitted_dataset.slices['test_mask']
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
batch = parser.batch
# model = GCN2_mask(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
#                    shared_weights, dropout)
model = GCN_mask(num_layers, dim_node, dim_hidden, num_classes)
ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN','GM_GCN_best.pth')
model.load_state_dict(torch.load(ckpt_path)['net'])
model.to(device)
# model = GAT_mask(num_layers, dim_node, 300, num_classes, heads = [7,4,1])
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GAT','GAT_100_best.pth')
# model.load_state_dict(torch.load(ckpt_path)['net'])
# model.to(device)
# dim_hidden = 300
# model = GCN_2l_mask(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
# model.to(device)
# check_checkpoints()
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
# model.load_state_dict(torch.load(ckpt_path)['state_dict'])
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GCN2','GCN2_best.pth')
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
# model.load_state_dict(torch.load(ckpt_path)['state_dict'])
# coff_size: 0.01
# coff_ent: 0.5
# coff_pred: 5 fidelity:0.16 tensor(0.1502) [0.04, 1.0, 1]
# coff_size: 0.035
# coff_ent: 0.5
# coff_pred: 1
# Fidelity: 0.1652
# coff_size: 0.033
# coff_ent: 2
# coff_pred: 1.5
# Fidelity: 0.2849
# Sparsity: 0.9605
# lr: 0.001


def find_motif(edge_index, node_idx, y, reserve):
    for i in range(edge_index.shape[-1]):
        if edge_index[0][i] == node_idx:
            if y[node_idx] in [1,2,3]:
                if y[edge_index[1][i]] in [1,2,3]:
                    if i not in reserve:
                        reserve.add(i)
                        reserve = reserve.union(find_motif(edge_index, edge_index[1][i], y, reserve))
            elif y[node_idx] in [5,6,7]:
                if y[edge_index[1][i]] in [5,6,7]:
                    if i not in reserve:
                        reserve.add(i)
                        reserve = reserve.union(find_motif(edge_index, edge_index[1][i], y, reserve))
        elif edge_index[1][i] == node_idx:
            if y[node_idx] in [1,2,3]:
                if y[edge_index[0][i]] in [1,2,3]:
                    if i not in reserve:
                        reserve.add(i)
                        reserve = reserve.union(find_motif(edge_index, edge_index[0][i], y, reserve))
            elif y[node_idx] in [5,6,7]:
                if y[edge_index[0][i]] in [5,6,7]:
                    if i not in reserve:
                        reserve.add(i)
                        reserve = reserve.union(find_motif(edge_index, edge_index[0][i], y, reserve))
    return reserve
try:
    motif = pk.load(open('Ba_Community_motif.plk','rb'))
except:
    data = dataset.data
#     motif = {}
#
#     for node_idx in tqdm(torch.where(((data.y != 0).int()) + ((data.y !=4).int()) == 2)[0].tolist()):
#         tmp = list(find_motif(data.edge_index, node_idx, data.y, set()))
#         if len(tmp) == 12:
#             motif[node_idx] = tmp
#     pk.dump(motif, open('Ba_shapes_motif.plk','wb'))
#     sys.exit()
# correct = 0
# fidelity = 0
# for i, node_idx in tqdm(enumerate(motif.keys()), total= len(list(motif.keys()))):
#     data = dataset.data
#     edge_index = data.edge_index[:,motif[node_idx]]
#     x_set = set()
#     x_set = x_set.union(set(edge_index[0,:]))
#     x_set = x_set.union(set(edge_index[1,:]))
#     x_set_r = set()
#     for e in x_set:
#             x_set_r.add(e.item())
#     # print(x_set_r)
#     # continue
#     # print(data.y[list(x_set_r)])
#     # continue
#     node_color = ['orange', 'red', 'green', 'blue', 'maroon', 'brown', 'darkslategray', 'paleturquoise']
#     colors = [node_color[i] for i in data.y[sorted(list(x_set_r))]]
#     # print(colors, node_idx)
#     # continue
#     # if node_idx != 350:
#     #     continue
#     # print(colors, node_idx)
#     # print(data.y[node_idx])
#     # sys.exit()
#     G = nx.Graph()
#     G.add_nodes_from(torch.unique(edge_index[0]).tolist())
#     for i, (u, v) in enumerate(edge_index.t().tolist()):
#         G.add_edge(u, v)
#     graph = G
#     pos = nx.kamada_kawai_layout(graph)
#     nx.draw_networkx_nodes(graph, pos,
#                            nodelist=list(graph.nodes()),
#                            node_color=colors,
#                            node_size=60)
#     # print(colors)
#     nx.draw_networkx_edges(graph, pos, width=1, edge_color='black', arrows=False)
#     node_idx_color = [node_color[data.y[node_idx]] ]
#     # print(node_idx_color)
#     nx.draw_networkx_nodes(graph, pos=pos,
#                            nodelist=[node_idx],
#                            node_color=node_idx_color,
#                            node_size=200)
#
#     plt.axis('off')
#     plt.savefig(f'fig/motifs/{node_idx}.jpg')
#     plt.show()
# sys.exit()
#     for e in x_set:
#         x_set_r.add(e.item())
#     with torch.no_grad():
#         a = model(data.x.cuda(), data.edge_index.cuda())[node_idx]
#         b = model(data.x.cuda(), edge_index.cuda())[node_idx]
#     if a.argmax(-1) == b.argmax(-1):
#         correct += 1
#     fidelity += abs(F.softmax(a, -1)[a.argmax(-1)]-F.softmax(b, -1)[a.argmax(-1)])
# #     print(a-b, F.softmax(model(data.x.cuda(), data.edge_index.cuda())[node_idx], dim = -1).argmax(-1), F.softmax(model(data.x.cuda(), edge_index.cuda())[node_idx], dim = -1).argmax(-1), data.y[node_idx])
# #     if a - b > 0.5:
# #         print(edge_index, data.y[list(x_set_r)])
# #         sys.exit()
# print('correct:',correct/(i+1))
# print('inv_fidelity:',fidelity/(i+1))
# sys.exit()
torch.backends.cudnn.benchmark = True

# tensor(0.1331) [0.34, 1, 1.5, 0.001]
# tensor(0.1334) [0.3, 2.5, 1, 0.001]
coff_sizes = [0.03]
coff_ents = [1]
coff_preds = [2]
lrs = [0.003]



best_spar = 0
best_parameters = []
# for coff_size, coff_ent, coff_pred, lr in tqdm(iterable= product(coff_sizes, coff_ents, coff_preds, lrs),
#                                                total= len(list(product(coff_sizes, coff_ents, coff_preds, lrs)))):
for coff_size, coff_ent, coff_pred, lr in product(coff_sizes, coff_ents, coff_preds, lrs):
    data = dataset[0]
    data.to(device)
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    explainer = PGExplainer(model, lr = lr, in_channels=3*dim_hidden,
                            device=device, explain_graph=False, num_hops = 2, epochs = 1000,
                            coff_size= coff_size, coff_ent= coff_ent, coff_pred = coff_pred).cuda()
    # explainer.train_explanation_network(data.cuda(), batch = batch)
    # torch.save(explainer.state_dict(), 'checkpoints/explainer/cora/pgexplainer_gcn_sub_1000epoch.pt')
    state_dict = torch.load('checkpoints/explainer/cora/pgexplainer_gcn_sub_1000epoch.pt')
    #
    explainer.load_state_dict(state_dict)


    data = dataset[0].cuda()
    # node_indices = torch.where(((data.y != 0).int()) + ((data.y !=4).int()) == 2)[0].tolist()
    # from dig.xgraph.method.pgexplainer import PlotUtils
    # plotutils = PlotUtils(dataset_name='ba_community')
    # with torch.no_grad():
    #     emb = explainer.model.get_emb(data.x, data.edge_index)
    # node_idx = 328
    # walks, masks, related_preds = \
    #             explainer(data, emb, node_idx=node_idx, y=data.y, top_k=6, sparsity = 0.5)
    #
    # explainer.visualization(data, edge_mask=masks[0], top_k=6, plot_utils=plotutils, node_idx=node_idx, vis_name = f'fig/pgexplainer_bacom_gcn100/pgexplainer_bacom_gcn100{node_idx}.pdf')
    # sys.exit()
    # for i in range(len(node_indices)):
    #     explainer = explainer.cuda()
    #     explainer.model = explainer.model.cuda()
    #     data=data.cuda()
    #     node_idx = node_indices[i]
    #     walks, masks, related_preds = \
    #         explainer(data, node_idx=node_idx, y=data.y, top_k=6)
    #     explainer.visualization(data, edge_mask=masks[0], top_k=6, plot_utils=plotutils, node_idx=node_idx, vis_name = f'fig/pgexplainer_bacom_gcn{node_idx}.pdf')
    # sys.exit()

# --- Create data collector and explanation processor ---

    x_collector = XCollector()

    ## Run explainer on the given model and dataset
    explainer.eval()
    a = 0.5
    b = 0.05
    c = []
    fidelitys = []
    sparsitys = []
    while a < 1:
        c.append(a)
        a += b
    # --- Set the Sparsity to 0.5
    # large_index = pk.load(open('large_subgraph_bacom.pk','rb'))['node_idx']
    # motif = pk.load(open('Ba_Community_motif.plk','rb'))
    data = dataset[0]
    subgraphs = {}
    explain_node_index_list = torch.where(data.test_mask)[0]
    for j, node_idx in tqdm(enumerate(explain_node_index_list), total= len(explain_node_index_list)):
        x, edge_index, y, subset, _ = explainer.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
        subgraphs[j] = {'x':x.cpu(), 'edge_index':edge_index.cpu(), 'new_node_idx':torch.where(subset == node_idx)[0].cpu(),
                        'subset':subset}

    data = data.to(device)
    emb = explainer.model.get_emb(data.x, data.edge_index)
    for explanation_confidence in c:
        with torch.no_grad():
            # indices = list(set(large_index).intersection(set(motif.keys())))
            spars = 0
            for j, node_idx in tqdm(enumerate(explain_node_index_list), total = len(explain_node_index_list)):
                # print(f'explain graph {i} node {node_idx}')

                subgraph = subgraphs[j]

                walks, masks, related_preds= \
                    explainer(data,emb = emb,explanation_confidence = explanation_confidence, node_idx=node_idx,
                              x = subgraph['x'], edge_index = subgraph['edge_index'], new_node_idx = subgraph['new_node_idx'],
                              subset = subgraph['subset'])
                # print(related_preds)
                # sys.exit()
                spars += related_preds[0]['sparsity']

                # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
                # obtain the result: x_processor(data, masks, x_collector)
            sparsity = spars/(j + 1)
            sparsitys.append(sparsity)
            print('hyper parameters:\n'
                  f'coff_size: {coff_size}\n'
                  f'coff_ent: {coff_ent}\n'
                  f'coff_pred: {coff_pred}')

            if sparsity > best_spar:
                best_spar = sparsity
                best_parameters = [coff_size, coff_ent, coff_pred, lr]
            print(f'Explanation_Confidence: {explanation_confidence:.2f}\n'
                  f'Sparsity: {sparsity:.4f}')

print(sparsitys)
# print(best_fidelity, best_parameters)
#
# print('fidelity: ',fidelitys, '\ninv_fidelity: ',inv_fidelitys)