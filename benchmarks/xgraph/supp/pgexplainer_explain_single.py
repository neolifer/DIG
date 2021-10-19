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
parser.add_argument('--model_name', default='GCN_nopre')
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
parser.add_argument('--batch', default=10)
parser = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




if parser.dataset_name not in  ['Cora','Pubmed']:
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
model = GM_GCN(n_layers = num_layers, input_dim = dim_node, hid_dim = dim_hidden, n_classes = num_classes)
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GM_GCN','GM_GCN_100_best.pth')

# model = GM_GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
#                 shared_weights)
ckpt_path = osp.join('checkpoints', parser.dataset_name.lower(), f'GM_{parser.model_name.split("_")[0]}',f'GM_{parser.model_name}_best.pth')
# ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN2','GCN2_best.pth')
model.load_state_dict(torch.load(ckpt_path)['net'])
model.to(device)

torch.backends.cudnn.benchmark = True

# tensor(0.1331) [0.34, 1, 1.5, 0.001]
# tensor(0.1334) [0.3, 2.5, 1, 0.001]
batch_size = 1
coff_sizes = [5e-3, 1e-2, 1e-3]
coff_ents = [1]
coff_preds = [1]
lrs = [0.0001, 0.001]



best_spar = 0
best_parameters = []
# for coff_size, coff_ent, coff_pred, lr in tqdm(iterable= product(coff_sizes, coff_ents, coff_preds, lrs),
#                                                total= len(list(product(coff_sizes, coff_ents, coff_preds, lrs)))):
sparsity_all = []
for coff_size, coff_ent, coff_pred, lr in product(coff_sizes, coff_ents, coff_preds, lrs):
    data = dataset.data
    data.to(device)
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    explainer = PGExplainer(model, lr = lr, in_channels=3*dim_hidden,
                            device=device, explain_graph=False, epochs = 100,
                            coff_size= coff_size, coff_ent= coff_ent, coff_pred = coff_pred, batch_size = batch_size).cuda()
    ## Run explainer on the given model and dataset

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
    data = dataset[0].to(explainer.device)
    # data = dataset.data.to(explainer.device)
    explain_node_index_list = torch.where(data.test_mask)[0]
    # explain_node_index_list = list(set(large_index).intersection(set(motif.keys())))
    subgraphs = {}
    try:
        subgraphs = torch.load(f'checkpoints/pgexplainer_sub/pgexplainer_{parser.dataset_name}_sub_test.pt')
    except:
        with torch.no_grad():
            for j, node_idx in tqdm(enumerate(explain_node_index_list), total= len(explain_node_index_list)):
                x, edge_index, y, subset, _ = explainer.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
                edge_index = add_remaining_self_loops(edge_index)[0]
                emb = explainer.model.get_emb(x, edge_index)
                new_node_idx = torch.where(subset == node_idx)[0]
                col, row = edge_index
                f1 = emb[col]
                f2 = emb[row]
                self_embed = emb[new_node_idx].repeat(f1.shape[0], 1)
                f12self = torch.cat([f1, f2], dim=-1)
                subgraphs[j] = {'x':x.cpu(), 'edge_index':edge_index.cpu(), 'new_node_idx':new_node_idx.cpu(),
                                'subset':subset, 'emb':f12self.cpu(), 'node_size': emb.shape[0], 'feature_dim':emb.shape[-1]}
        torch.save(subgraphs, f'checkpoints/pgexplainer_sub/pgexplainer_{parser.dataset_name}_sub_test.pt')
    subgraphs = torch.load(f'checkpoints/pgexplainer_sub/pgexplainer_{parser.dataset_name}_sub_test.pt')

    for _ in range(1):
        # indices = list(set(large_index).intersection(set(motif.keys())))
        spars = [0 for _ in range(len(c))]
        for j, node_idx in tqdm(enumerate(explain_node_index_list), total = len(explain_node_index_list)):

            torch.manual_seed(42)
            random.seed(0)
            np.random.seed(0)
            explainer = PGExplainer(model, lr = lr, in_channels=2*dim_hidden,
                                    device=device, explain_graph=False, epochs = 1000,
                                    coff_size= coff_size, coff_ent= coff_ent, coff_pred = coff_pred, batch_size = batch_size).cuda()
            # print(f'explain graph {i} node {node_idx}')
            subgraph = subgraphs[j]
            print(subgraph['edge_index'].shape[-1])
            related_preds= \
                explainer.train_explain_single(emb = subgraph['emb'],explanation_confidence = c, node_idx=node_idx,
                          x = subgraph['x'], edge_index = subgraph['edge_index'], new_node_idx = subgraph['new_node_idx'],
                          subset = subgraph['subset'], node_size = subgraph['node_size'],
                          feature_dim = subgraph['feature_dim'])
            # print(related_preds)
            # sys.exit()
            for i in range(len(c)):
                spars[i] += related_preds[i]['sparsity']

            # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
            # obtain the result: x_processor(data, masks, x_collector)
        for i in range(len(c)):
            sparsity = spars[i]/(j + 1)
            sparsitys.append(sparsity)
            print('hyper parameters:\n'
                  f'coff_size: {coff_size}\n'
                  f'coff_ent: {coff_ent}\n'
                  f'coff_pred: {coff_pred}')

            if sparsity > best_spar:
                best_spar = sparsity
                best_parameters = [coff_size, coff_ent, coff_pred, lr]
            print(f'Explanation_Confidence: {c[i]:.2f}\n'
                  f'Sparsity: {sparsity:.4f}')
            sparsity_all.append(sparsitys)

print(sparsity_all)
# print(best_fidelity, best_parameters)
#
# print('fidelity: ',fidelitys, '\ninv_fidelity: ',inv_fidelitys)