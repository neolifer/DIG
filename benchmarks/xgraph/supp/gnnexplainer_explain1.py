import sys
sys.path.append('../../..')
from tqdm import tqdm
from torch_geometric.utils.loop import add_self_loops, remove_self_loops,add_remaining_self_loops
sys.path.append('../../../..')
from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.models import *
from dig.xgraph.method.shapley import GnnNets_GC2value_func, GnnNets_NC2value_func, gnn_score
from torch_geometric.nn import GNNExplainer
import torch.nn.functional as F
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
import os.path as osp
import os
import argparse
from PGExplainer.load_dataset import get_dataset, get_dataloader
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import k_hop_subgraph, to_networkx
from copy import copy
from math import sqrt
from typing import Optional
from dig.xgraph.method.pgexplainer import k_hop_subgraph_with_default_whole_graph, calculate_selected_nodes
from dig.xgraph.utils import *
import math
import pickle as pk
from torch_geometric.datasets import CitationFull, Planetoid
import torch_geometric.transforms as T
import yaml
from yaml import SafeLoader

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



if parser.dataset_name not in  ['Cora','Pubmed','Citeseer']:
    dataset = get_dataset(parser)
else:
    if parser.model_name == 'GCN2':
        dataset = Planetoid('./datasets', parser.dataset_name,split="public", transform = T.NormalizeFeatures())
    else:
        dataset = Planetoid('./datasets', parser.dataset_name,split="public")
dataset.data.x = dataset.data.x.to(torch.float32)
# dataset.data.x = dataset.data.x[:, :1]
# dataset.data.y = dataset.data.y[:, 2]
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
# num_targets = dataset.num_classes
num_classes = dataset.num_classes
print(parser.model_name, parser.dataset_name, 'GNNExplainer')


def check_checkpoints(root='./'):
    if osp.exists(osp.join(root, 'checkpoints')):
        return
    url = ('https://github.com/divelab/DIG_storage/raw/main/xgraph/checkpoints.zip')
    path = download_url(url, root)
    extract_zip(path, root)
    os.unlink(path)


model_level = parser.model_level
dim_hidden = int(parser.dim_hidden)
alpha = parser.alpha
theta=parser.theta
num_layers= int(parser.num_layers)
shared_weights=parser.shared_weights
dropout=parser.dropout
if 'GCN2' in parser.model_name:
    config = yaml.load(open('config.yaml'), Loader=SafeLoader)[parser.dataset_name]
    alpha = config['alpha']
    theta = config['lambda']
    num_layers = config['num_layers']
    dim_hidden = config['dim_hidden']
# model = GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
# shared_weights, dropout)
# model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
# model = GM_GCN(num_layers, dim_node, dim_hidden, num_classes)
# model2 = GAT(num_layers, dim_node, 300, num_classes, heads = [7,4,1])
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GAT','GAT_100_best.pth')
# model2.load_state_dict(torch.load(ckpt_path)['net'])
# model2.to(device)
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GIN_2l', '0', 'GIN_2l_best.ckpt')
# model.load_state_dict(torch.load(ckpt_path)['state_dict'])
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN2','GCN2_best.pth')
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l','GCN_2l_best.pth')
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GM_GCN','GM_GCN_100_best.pth')
# model = GM_GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
#                 shared_weights)
# ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN2','GCN2_best.pth')
if 'GCN2' in parser.model_name:
    model = GM_GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
                      shared_weights, dropout)
elif 'GAT' in parser.model_name:
    if not parser.dataset_name == 'Pubmed':
        model = GM_GAT(num_layers, dim_node, dim_hidden, num_classes, dropout, heads = [8,1])
    else:
        model = GM_GAT(num_layers, dim_node, dim_hidden, num_classes, dropout, heads = [8,8])
else:
    model = GM_GCN(n_layers = num_layers, input_dim = dim_node, hid_dim = dim_hidden, n_classes = num_classes)
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GM_GCN','GM_GCN_100_best.pth')
ckpt_path = osp.join('checkpoints', parser.dataset_name.lower(), f'GM_{parser.model_name.split("_")[1]}',f'{parser.model_name}_best.pth')
model.load_state_dict(torch.load(ckpt_path)['net'])
from dig.xgraph.method import GNNExplainer
# explainer = GNNExplainer(model, epochs=100, lr=0.01, explain_graph=False)
explainer = GNNExplainer(model, epochs=100, lr=0.01, explain_graph=False)
explainer.model.set_get_vertex(False)
# edge_index = dataset[0].edge_index
# dist_dict = {}
# for i, (u, v) in enumerate(edge_index.t().tolist()):
#     if u == 0 and v != 0:
#         dist_dict[(u,v)] = 100
#     elif u != 0 and v == 0:
#         dist_dict[(u,v)] = 100
#     elif u == 0 and v == 0:
#         dist_dict[(u,v)] = 0.01
#     else:
#         dist_dict[(u,v)] = 0.25
# print(len(list(dist_dict.keys())))

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
#         hard_mask, new_x, new_y, edge_masks, _= \
#             explainer2(data.x, data.edge_index, sparsity=sparsity, num_classes=num_classes, node_idx=node_idx,y = data.y, control_sparsity = False)
#         # print(edge_masks[data.y[node_idx]].shape, add_self_loops(data.edge_index)[0].shape)
#         # sys.exit()
#         edge_mask = edge_masks[data.y[node_idx]]
#         self_loop = add_self_loops(data.edge_index)[0]
#         new_edge_index = self_loop[:, hard_mask]
#         edge_mask = edge_mask[hard_mask]
#         real_gate = []
#         for i in range(new_edge_index.shape[-1]):
#             temp = new_edge_index[:, i]
#             if temp[0] != temp[1]:
#                 real_gate.append(i)
#         new_edge_index = remove_self_loops(new_edge_index)[0]
#         edge_mask = edge_mask[real_gate]
#         new_data = Data(x = new_x, edge_index= new_edge_index, y= new_y)
#         visualize(new_data, edge_mask=edge_mask, top_k=6, plot_utils=plotutils, node_idx= node_idx, vis_name = f'fig/gnnexplainer_gcn100/ba_com/motif/{node_idx}.pdf', dist_dict = dist_dict)
#         plt.show()
# sys.exit()
a = 0.5
b = 0.05
c = []
while a < 1:
    c.append(a)
    a += b
# --- Set the Sparsity to 0.5

sparsitys = []


# large_index = pk.load(open('large_subgraph_bacom.pk','rb'))['node_idx']
# motif = pk.load(open('Ba_shapes_motif.plk','rb'))
# # for key in motif:
# #     print(motif[key])
# #     sys.exit()
#
# in_motif = []
# ec = []
# for _ in range(1):
#     # node_indices = list(set(large_index).intersection(set(motif.keys())))
#
#     data = dataset[0]
# #     # node_indices = list(set(large_index).intersection(set(motif.keys())))
#     explain_node_index_list = list(motif.keys())
# #     if parser.dataset_name == 'BA_Community' or parser.dataset_name == 'BA_shapes':
# #         explain_node_index_list = pk.load(open(f'{parser.dataset_name}_explanation_node_list.pk','rb'))
# #     elif parser.dataset_name in ['Cora','Pubmed','Citeseer']:
# #         explain_node_index_list = torch.where(data.test_mask)[0]
# #     else:
# #         explain_node_index_list = [i for i in range(data.y.shape[0]) if data.y[i] != 0]
# #     spar = [0 for e in c]
#     no_count = 0
#     for j, node_idx in tqdm(enumerate(explain_node_index_list), total = len(explain_node_index_list)):
#         import random
#         import numpy as np
#         torch.manual_seed(42)
#         random.seed(0)
#         np.random.seed(0)
#         data.to(device)
#         explainer = GNNExplainer(model, epochs=100, lr=0.01, explain_graph=False)
#         # try:
#         _, _, _, mask, related_preds = \
#             explainer(data.x, data.edge_index, num_classes=num_classes, node_idx=node_idx,
#                       y = data.y, evaluation_confidence = c,control_sparsity = False, motif = motif[node_idx])
#         # except:
#         #     no_count += 1
#         #     continue
#         in_motif.append(related_preds['in_motif'][0])
#         ec.append(related_preds['evaluation_confidence'])
#
# #         for i in range(len(spar)):
# #             spar[i] += related_preds[i]
# # for i in range(len(spar)):
# #     sparsity = spar[i]/(j + 1 - no_count)
# #     sparsitys.append(sparsity)
# #     print('GCN:',f'Evaluation Confidence: {c[i]:.2f}\n'
# #           f'Sparsity: {sparsity:.4f}\n')
# #
# result = {'ec':ec,'in_motif':in_motif,'dataset':parser.dataset_name}
# # result = {'sparsity':sparsitys, 'method':'gnnexplainer'}
# pk.dump(result, open(f'in_motif_{result["dataset"]}.pk','wb'))
# # print(sparsitys)
# # pk.dump(temp, open('results/bacom/gnnexplainer_gcn100_bacom_result.pk','wb'))
# # pk.dump({'node_idx':large_index}, open('large_subgraph_bacom.pk','wb'))
# sys.exit()

# for _ in range(1):
#     # node_indices = list(set(large_index).intersection(set(motif.keys())))
#     data = dataset[0]
#     print(data.y.unique())
#     sys.exit()
#     # node_indices = list(set(large_index).intersection(set(motif.keys())))
#     # node_indices = list(motif.keys())
#     if parser.dataset_name == 'BA_Community' or parser.dataset_name == 'BA_shapes':
#         explain_node_index_list = pk.load(open(f'{parser.dataset_name}_explanation_node_list.pk','rb'))
#     elif parser.dataset_name in ['Cora','Pubmed','Citeseer']:
#         explain_node_index_list = torch.where(data.test_mask)[0]
#     else:
#         explain_node_index_list = [i for i in range(data.y.shape[0]) if data.y[i] != 0]
#     sparsities = []
#     fidelities = []
#     accs = []
#     s = 0
#     while s < 0.45:
#         fidelities.append(0)
#         accs.append(0)
#         sparsities.append(0.5+s)
#         s += 0.05
#     no_count = 0
#     for j, node_idx in tqdm(enumerate(explain_node_index_list), total = len(explain_node_index_list)):
#         import random
#         import numpy as np
#         torch.manual_seed(42)
#         random.seed(0)
#         np.random.seed(0)
#         data.to(device)
#         explainer = GNNExplainer(model, epochs=100, lr=0.01, explain_graph=False)
#         try:
#             _, _, new_edge_index, mask, related_preds = \
#                 explainer(data.x, data.edge_index, num_classes=num_classes, node_idx=node_idx,
#                           y = data.y[node_idx], evaluation_confidence = c,control_sparsity = False)
#             # edge_count.append(new_edge_index.shape[-1])
#             if new_edge_index.shape[-1] <= 40 and new_edge_index.shape[-1] >= 20:
#                 print(new_edge_index.shape[-1],node_idx.item(), data.y[node_idx].item())
#         except:
#             no_count += 1
#             continue
#         for i in range(len(sparsities)):
#             fidelities[i] += related_preds['fidelity'][i]
#             accs[i] += related_preds['acc'][i]
# print(edge_count)
# feds = 0
# accus = 0
# for i in range(len(sparsities)):
#     sparsity = sparsities[i]
#     fidelity = fidelities[i]/(j-no_count)
#     fidelities[i] = float(f'{fidelity.item():.4f}')
#     acc = accs[i]/(j-no_count)
#     accs[i] = float(f'{acc:.4f}')
#     feds += fidelity
#     accus += acc
    # print(f'{parser.model_name.split("_")[1]}:',
    #         f'Sparsity: {sparsity:.2f}\n',
    #         f'fidelity: {fidelity:.4f}\n',
    #         f'acc: {acc:.4f}\n',
    #       )
# print('fidelity:',fidelities, 'acc:',accs)
# print(f'avg_fidelity:{feds/len(sparsities):.4f}, avg_acc:{accus/len(sparsities):.4f}')
# result = {'fidelity':fidelity,'acc':accs, 'method':'gnnexplainer','model':f'{parser.model_name.split("_")[1]}','dataset':f'{parser.dataset_name}'}
# result = {'sparsity':sparsitys, 'method':'gnnexplainer'}
# pk.dump(result, open(f'results_fidelity_{result["method"]}_{parser.dataset_name}.pk','wb'))
# print(sparsitys)
class_dict = {}
for i in range(7):
    class_dict[i] = [0,0,0,0,0,0,0]
for _ in range(1):
    # node_indices = list(set(large_index).intersection(set(motif.keys())))
    data = dataset[0]
    # node_indices = list(set(large_index).intersection(set(motif.keys())))
    # node_indices = list(motif.keys())
    if parser.dataset_name == 'BA_Community' or parser.dataset_name == 'BA_shapes':
        explain_node_index_list = pk.load(open(f'{parser.dataset_name}_explanation_node_list.pk','rb'))
    elif parser.dataset_name in ['Cora','Pubmed','Citeseer']:
        explain_node_index_list = torch.where(data.test_mask)[0]
    else:
        explain_node_index_list = [i for i in range(data.y.shape[0]) if data.y[i] != 0]
    sparsities = []
    fidelities = []
    accs = []
    s = 0
    while s < 0.45:
        fidelities.append(0)
        accs.append(0)
        sparsities.append(0.5+s)
        s += 0.05
    no_count = 0
    for j, node_idx in tqdm(enumerate(explain_node_index_list), total = len(explain_node_index_list)):
        import random
        import numpy as np
        torch.manual_seed(42)
        random.seed(0)
        np.random.seed(0)
        data.to(device)
        explainer = GNNExplainer(model, epochs=100, lr=0.01, explain_graph=False)
        # try:
        label, class_count = \
            explainer(data.x, data.edge_index, num_classes=num_classes, node_idx=node_idx,
                      y = data.y[node_idx], evaluation_confidence = c,control_sparsity = False,data = data)
        # edge_count.append(new_edge_index.shape[-1])
        for i in range(7):
            class_dict[label][i] += class_count[i]
        # except:
        #     no_count += 1
        #     continue

print(class_dict)
