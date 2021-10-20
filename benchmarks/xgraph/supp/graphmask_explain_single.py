import sys

import tqdm

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
from matplotlib import pyplot as plt
import seaborn
from torch_geometric.datasets import CitationFull, Planetoid
import torch_geometric.transforms as T
import pickle as pk
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN2', dest='gnn models')
parser.add_argument('--model_name', default='GCN_100_nopre')
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
parser.add_argument('--use_baseline', default=False)
parser = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


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





model_level = parser.model_level
dim_hidden = int(parser.dim_hidden)
alpha = parser.alpha
theta=parser.theta
num_layers= int(parser.num_layers)
shared_weights=parser.shared_weights
dropout=parser.dropout

# model = GCN2_mask(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
#                    shared_weights, dropout)
# model = GM_GCN(n_layers = num_layers, input_dim = dim_node, hid_dim = dim_hidden, n_classes = num_classes)

# model = GM_GAT(num_layers, dim_node, 300, num_classes, heads = [7,4,1])
# model.to(device)
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GAT','GAT_100_best.pth')
# model.load_state_dict(torch.load(ckpt_path)['net'])

model = GM_GCN(n_layers = num_layers, input_dim = dim_node, hid_dim = dim_hidden, n_classes = num_classes)
# model = GM_GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
#                 shared_weights)
model.to(device)
ckpt_path = osp.join('checkpoints', parser.dataset_name.lower(), f'GM_GCN',f'{parser.model_name}_best.pth')# model.load_state_dict(torch.load(ckpt_path)['net'])

# ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN','GM_GCN_nopre_best.pth')
model.load_state_dict(torch.load(ckpt_path)['net'])










# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN2','GCN2_best.pth')
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l','GCN_2l_best.pth')
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GM_GCN','GM_GCN_100_best.pth')

# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l', '0', 'GCN_2l_best.ckpt')
# model.load_state_dict(torch.load(ckpt_path)['state_dict'])

from itertools import product
from dig.xgraph.method import GraphMaskExplainer, GraphMaskAdjMatProbe
message_dims = [dim_hidden for _ in range(num_layers)]
hidden_dims = [dim_hidden for _ in range(num_layers)]
vertex_dims = [3*dim_hidden]+[3*dim_hidden for _ in range(num_layers - 1)]
GraphMask = GraphMaskAdjMatProbe(vertex_dims, message_dims, num_classes, hidden_dims)
model.cuda()
GraphMask.cuda()
allowance =  0.2
penalty_scalings = [0.001]
# penalty_scalings = [10]
entropy_scales = [1]
allowances = [0.05]
# allowances = [0.03]
lr1s = [3e-3]
lr2s = [1e-4]
best_parameters = None
best_spar = 1
data = dataset[0]
# large_index = pk.load(open('large_subgraph_bacom.pk','rb'))['node_idx']
# motif = pk.load(open('Ba_Community_motif.plk','rb'))
# explain_node_index_list = list(set(large_index).intersection(set(motif.keys())))
# explain_node_index_list = motif.keys()

subgraphs = {}
a = 0.50
b = 0.05
c = []
while a < 1:
    c.append(a)
    a += b
if parser.dataset_name == 'BA_Community' or parser.dataset_name == 'BA_shapes':
    explain_node_index_list = pk.load(open(f'{parser.dataset_name}_explanation_node_list.pk','rb'))
elif parser.dataset_name in ['Cora','Pubmed','Citeseer']:
    explain_node_index_list = torch.where(data.test_mask)[0]
else:
    explain_node_index_list = [i for i in range(data.y.shape[0]) if data.y[i] != 0]
# explain_node_index_list = pk.load(open(f'{parser.dataset_name}_exclude_nodes.pk','rb'))
# explain_node_index_list = pk.load(open(f'{parser.dataset_name}_explanation_node_list.pk','rb'))
for explain_node_index_list in [explain_node_index_list]:
    for penalty_scaling, entropy_scale, allowance,lr1, lr2 in product(penalty_scalings, entropy_scales,allowances, lr1s, lr2s):
        sparsitys = []
        explainer = GraphMaskExplainer(model, GraphMask, epoch = 100, penalty_scaling = penalty_scaling,
                                       entropy_scale = entropy_scale,allowance = allowance, lr1 =lr1, lr2= lr2, batch_size = len(explain_node_index_list))
        try:
            subgraphs = torch.load(f'checkpoints/graphmask_sub/graphmask_{parser.dataset_name}_sub_test.pt')
            # if not len(subgraphs.keys()) == len(explain_node_index_list):
            #     raise Exception
        except:
            for j, node_idx in tqdm.tqdm(enumerate(explain_node_index_list), total= len(explain_node_index_list)):
                x, edge_index, y, subset, _ = explainer.get_subgraph(node_idx, data.x, data.edge_index,data.y)
                subgraphs[j] = {'x':x.cpu(), 'edge_index':edge_index.cpu(), 'new_node_idx':torch.where(subset == node_idx)[0].cpu(), 'y':y}
            torch.save(subgraphs,f'checkpoints/graphmask_sub/graphmask_{parser.dataset_name}_sub_test.pt')
        subgraphs = torch.load(f'checkpoints/graphmask_sub/graphmask_{parser.dataset_name}_sub_test.pt')
        spars = [0 for _ in range(len(c))]
        no_count = 0
        for j, node_idx in tqdm.tqdm(enumerate(explain_node_index_list), total= len(explain_node_index_list)):


            import random
            import numpy as np
            torch.manual_seed(42)
            random.seed(0)
            np.random.seed(0)
            GraphMask = GraphMaskAdjMatProbe(vertex_dims, message_dims, num_classes, hidden_dims)
            GraphMask.cuda()
            explainer = GraphMaskExplainer(model, GraphMask, epoch = 200, penalty_scaling = penalty_scaling,
                                           entropy_scale = entropy_scale,allowance = allowance, lr1 =lr1, lr2= lr2, batch_size = len(explain_node_index_list)
                                           , use_baseline = parser.use_baseline)
            subgraph = subgraphs[j]
            if subgraph['edge_index'].shape[-1] <= 1:
                no_count += 1
                continue
            related_preds = explainer.train_explain_single(
                x = subgraph['x'], edge_index = subgraph['edge_index'], new_node_idx = subgraph['new_node_idx'])
            # sys.exit()
            for i in range(len(c)):
                spars[i] += related_preds[i]['sparsity']
        for i in range(len(c)):
            sparsity = spars[i]/(j + 1 - no_count)
            # sparsity = spars/(j + 1)
            sparsitys.append(sparsity)
            if sparsity < best_spar:
                best_parameters = [penalty_scaling, entropy_scale, allowance,lr1, lr2]
                best_spar = sparsity
            # print(f'parameters:{[penalty_scaling, entropy_scale, allowance,lr1, lr2]}\n'
            #       f'explanation_confidence: {c[i]:.2f}\n'
            #       f'Sparsity: {sparsity:.4f}')
        print('evaluation_confidence: ',c)
        print('sparsity: ', sparsitys)
        result = {'sparsity':sparsitys, 'parameters':best_parameters, 'method':'graphmask_single'}
        pk.dump(result, open(f'results{result["method"]}.pk','wb'))
print('best parameters: ', best_parameters,
      '\nbest inverse fidelity:', best_spar)