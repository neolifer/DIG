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
import yaml
from yaml import SafeLoader

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN2', dest='gnn models')
parser.add_argument('--model_name', default='GCN')
parser.add_argument('--model_level', default='node')
parser.add_argument('--dim_hidden', default=256)
parser.add_argument('--alpha', default=0.1)
parser.add_argument('--theta', default=0.6)
parser.add_argument('--num_layers', default=32)
parser.add_argument('--shared_weights', default=False)
parser.add_argument('--dropout', default=0.1)
parser.add_argument('--dataset_dir', default='./datasets/')
parser.add_argument('--dataset_name', default='Cora')
parser.add_argument('--epoch', default=1000)
parser.add_argument('--save_epoch', default=10)
parser.add_argument('--lr', default=0.01)
parser.add_argument('--wd1', default=1e-3)
parser.add_argument('--wd2', default=1e-5)
parser.add_argument('--batch', default=10)
parser = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




if parser.dataset_name not in  ['Cora','Pubmed','Citeseer']:
    dataset = get_dataset(parser)
else:
    if parser.model_name == 'GCN2':
        dataset = Planetoid('./datasets', parser.dataset_name,split="public", transform = T.NormalizeFeatures())
    else:
        dataset = Planetoid('./datasets', parser.dataset_name,split="public")
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

print(parser.model_name, parser.dataset_name, 'PGExplainer')


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
dropout= int(parser.dropout)
batch = int(parser.batch)
if 'GCN2' in parser.model_name:
    config = yaml.load(open('config.yaml'), Loader=SafeLoader)[parser.dataset_name]
    alpha = config['alpha']
    theta = config['lambda']
    num_layers = config['num_layers']
    dim_hidden = config['dim_hidden']
# model = GCN2_mask(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
#                    shared_weights, dropout)
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

# model = GM_GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
#                 shared_weights)
# ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN','GM_GCN_nopre_best.pth')
# ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN2','GCN2_best.pth')
ckpt_path = osp.join('checkpoints', parser.dataset_name.lower(), f'GM_{parser.model_name.split("_")[1]}',f'{parser.model_name}_best.pth')
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
batch_size = 1
coff_sizes = [6e-3]
coff_ents = [0.5]
coff_preds = [2] if parser.dataset_name != 'Pubmed' else [1]
lrs = [0.003]

data = dataset[0]
explain_node_index_list = torch.where(data.test_mask)[0]
best_spar = 0
best_parameters = []
# for coff_size, coff_ent, coff_pred, lr in tqdm(iterable= product(coff_sizes, coff_ents, coff_preds, lrs),
#                                                total= len(list(product(coff_sizes, coff_ents, coff_preds, lrs)))):
for conv in model.convs:
    conv.require_sigmoid = False
for coff_size, coff_ent, coff_pred, lr in product(coff_sizes, coff_ents, coff_preds, lrs):

    data.to(device)
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    if 'GAT' in parser.model_name:
        if parser.dataset_name == 'Pubmed':
            explainer = PGExplainer(model, lr = lr, in_channels=3*8*dim_hidden,
                                    device=device, explain_graph=False, epochs = 100,
                                    coff_size= coff_size, coff_ent= coff_ent, coff_pred = coff_pred, batch_size = batch_size).cuda()
        else:
            explainer = PGExplainer(model, lr = lr, in_channels=3*dim_hidden,
                                    device=device, explain_graph=False, epochs = 100,
                                    coff_size= coff_size, coff_ent= coff_ent, coff_pred = coff_pred, batch_size = batch_size).cuda()
    else:
        explainer = PGExplainer(model, lr = lr, in_channels=3*dim_hidden,
                                device=device, explain_graph=False, epochs = 100,
                                coff_size= coff_size, coff_ent= coff_ent, coff_pred = coff_pred, batch_size = batch_size).cuda()
    # explainer.train_explanation_network(data.cuda(), batch = batch, explain_node_index_list = explain_node_index_list)
    # torch.save(explainer.state_dict(), 'checkpoints/explainer/cora/pgexplainer_gat_sub_1000epoch_confirm_nopre.pt')
    # state_dict = torch.load(f'checkpoints/explainer/{parser.dataset_name.lower()}/pgexplainer_gat_sub_1000epoch_confirm_nopre.pt')
    # # torch.cuda.empty_cache()
    # model = GCN_mask(num_layers, dim_node, dim_hidden, num_classes)
    # ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN','GM_GCN_best.pth')
    # model.load_state_dict(torch.load(ckpt_path)['net'])
    # model.to(device)

    # explainer = PGExplainer(model, lr = lr, in_channels=3*dim_hidden,
    #                         device=device, explain_graph=False, num_hops = 2, epochs = 100,
    #                         coff_size= coff_size, coff_ent= coff_ent, coff_pred = coff_pred, batch_size = batch_size).cuda()
    state_dict = torch.load('checkpoints/explainer/cora/pgexplainer_gat_sub_1000epoch_confirm_nopre.pt')
    explainer.load_state_dict(state_dict)



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
    # data = dataset[0].to(explainer.device)
    data = dataset[0].to(explainer.device)
    # explain_node_index_list = list(set(large_index).intersection(set(motif.keys())))
    subgraphs = {}
    explain_node_index_list = torch.where(data.test_mask)[0]
    with torch.no_grad():
        for j, node_idx in tqdm(enumerate(explain_node_index_list), total= len(explain_node_index_list)):
        # for j, node_idx in enumerate(explain_node_index_list):
            x, edge_index, y, subset, _ = explainer.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
            edge_index = add_remaining_self_loops(edge_index)[0]
            emb = explainer.model.get_emb(x, edge_index)
            new_node_idx = torch.where(subset == node_idx)[0]
            col, row = edge_index
            f1 = emb[col]
            f2 = emb[row]
            self_embed = emb[new_node_idx].repeat(f1.shape[0], 1)
            f12self = torch.cat([f1, f2, self_embed], dim=-1)
            subgraphs[j] = {'x':x.cpu(), 'edge_index':edge_index.cpu(), 'new_node_idx':new_node_idx.cpu(),
                            'subset':subset, 'emb':f12self.cpu(), 'node_size': emb.shape[0], 'feature_dim':emb.shape[-1],'y':y}

    y = data.y
    data = None
#     for _ in range(1):
#         with torch.no_grad():
#             # indices = list(set(large_index).intersection(set(motif.keys())))
#             spars = [0 for _ in range(len(c))]
#             real_j = 0
#             for j, node_idx in tqdm(enumerate(explain_node_index_list), total = len(explain_node_index_list)):
#                 # print(f'explain graph {i} node {node_idx}')
#
#                 subgraph = subgraphs[j]
#
#                 walks, masks, related_preds= \
#                     explainer(emb = subgraph['emb'],explanation_confidence = c, node_idx=node_idx,
#                               x = subgraph['x'], edge_index = subgraph['edge_index'], new_node_idx = subgraph['new_node_idx'],
#                               subset = subgraph['subset'], node_size = subgraph['node_size'],
#                               feature_dim = subgraph['feature_dim'])
#                 # print(related_preds)
#                 # sys.exit()
#                 if related_preds is not None:
#                     real_j += 1
#                     for i in range(len(c)):
#                         spars[i] += related_preds[i]['sparsity']
#                 else:
#                     continue
#                 # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
#                 # obtain the result: x_processor(data, masks, x_collector)
#         for i in range(len(c)):
#             sparsity = spars[i]/real_j
#             sparsitys.append(sparsity)
#             print('hyper parameters:\n'
#                   f'coff_size: {coff_size}\n'
#                   f'coff_ent: {coff_ent}\n'
#                   f'coff_pred: {coff_pred}')
#
#             if sparsity > best_spar:
#                 best_spar = sparsity
#                 best_parameters = [coff_size, coff_ent, coff_pred, lr]
#             print(f'Explanation_Confidence: {c[i]:.2f}\n'
#                   f'Sparsity: {sparsity:.4f}')
#
# print(sparsitys)
# print(best_fidelity, best_parameters)
#
# print('fidelity: ',fidelitys, '\ninv_fidelity: ',inv_fidelitys)
#     for _ in range(1):
#         with torch.no_grad():
#             # indices = list(set(large_index).intersection(set(motif.keys())))
#             sparsities = []
#             fidelities = []
#             accs = []
#             s = 0
#             while s < 0.45:
#                 fidelities.append(0)
#                 accs.append(0)
#                 sparsities.append(0.5+s)
#                 s += 0.05
#             no_count = 0
#             for j, node_idx in tqdm(enumerate(explain_node_index_list), total = len(explain_node_index_list)):
#                 # print(f'explain graph {i} node {node_idx}')
#
#                 subgraph = subgraphs[j]
#
#                 walks, masks, related_preds= \
#                     explainer(emb = subgraph['emb'],explanation_confidence = c, node_idx=node_idx,
#                               x = subgraph['x'], edge_index = subgraph['edge_index'], new_node_idx = subgraph['new_node_idx'],
#                               subset = subgraph['subset'], node_size = subgraph['node_size'],
#                               feature_dim = subgraph['feature_dim'],y=y[node_idx])
#                 # print(related_preds)
#                 # sys.exit()
#                 if subgraph['edge_index'].shape[-1] <= 1:
#                     no_count += 1
#                     continue
#                 # sys.exit()
#                 for i in range(len(sparsities)):
#                     fidelities[i] += related_preds['fidelity'][i]
#                     accs[i] += related_preds['acc'][i]
#     feds = 0
#     accus = 0
#     for i in range(len(sparsities)):
#         sparsity = sparsities[i]
#         fidelity = fidelities[i]/(j-no_count)
#         fidelities[i] = float(f'{fidelity:.4f}')
#         acc = accs[i]/(j-no_count)
#         accs[i] = float(f'{acc:.4f}')
#         feds += fidelity
#         accus += acc
#         # print(f'{parser.model_name.split("_")[1]}:',
#         #         f'Sparsity: {sparsity:.2f}\n',
#         #         f'fidelity: {fidelity:.4f}\n',
#         #         f'acc: {acc:.4f}\n',
#         #       )
#     print('fidelity:',fidelities, 'acc:',accs)
#     print(f'avg_fidelity:{feds/len(sparsities):.4f}, avg_acc:{accus/len(sparsities):.4f}')
#     result = {'fidelity':fidelity,'acc':accs, 'method':'pgexplainer','model':f'{parser.model_name.split("_")[1]}','dataset':f'{parser.dataset_name}'}
#     # result = {'sparsity':sparsitys, 'method':'gnnexplainer'}
#     pk.dump(result, open(f'results_fidelity_{result["method"]}_{parser.dataset_name}.pk','wb'))
    for _ in range(1):
        class_dict = {}
        for i in range(7):
            class_dict[i] = [0,0,0,0,0,0,0]
        with torch.no_grad():
            # indices = list(set(large_index).intersection(set(motif.keys())))
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
                # print(f'explain graph {i} node {node_idx}')

                subgraph = subgraphs[j]

                label, class_count= \
                    explainer(emb = subgraph['emb'],explanation_confidence = c, node_idx=node_idx,
                              x = subgraph['x'], edge_index = subgraph['edge_index'], new_node_idx = subgraph['new_node_idx'],
                              subset = subgraph['subset'], node_size = subgraph['node_size'],
                              feature_dim = subgraph['feature_dim'],y=y[node_idx],new_y = subgraph['y'])
                for i in range(7):
                    class_dict[label][i] += class_count[i]
print(class_dict)