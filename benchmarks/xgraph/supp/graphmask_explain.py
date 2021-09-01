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


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='GCN2', dest='gnn models')
parser.add_argument('--model_name', default='GCN2')
parser.add_argument('--model_level', default='node')
parser.add_argument('--dim_hidden', default=64)
parser.add_argument('--alpha', default=0.1)
parser.add_argument('--theta', default=0.5)
parser.add_argument('--num_layers', default=64)
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
torch.manual_seed(42)
import random
random.seed(0)
import numpy as np
np.random.seed(0)



def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

# def split_dataset(dataset, num_classes):
#     indices = []
#     train_percent = 0.7
#     for i in range(num_classes):
#         index = (dataset.data.y == i).nonzero().view(-1)
#         index = index[torch.randperm(index.size(0))]
#         indices.append(index)
#
#     train_index = torch.cat([i[:int(len(i) * train_percent)] for i in indices], dim=0)
#
#     rest_index = torch.cat([i[int(len(i) * train_percent):] for i in indices], dim=0)
#     rest_index = rest_index[torch.randperm(rest_index.size(0))]
#
#     dataset.data.train_mask = index_to_mask(train_index, size=dataset.data.num_nodes)
#     dataset.data.val_mask = index_to_mask(rest_index[:len(rest_index) // 2], size=dataset.data.num_nodes)
#     dataset.data.test_mask = index_to_mask(rest_index[len(rest_index) // 2:], size=dataset.data.num_nodes)
#
#     dataset.data, dataset.slices = dataset.collate([dataset.data])
#
#     return dataset
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

# splitted_dataset = split_dataset(dataset, num_classes)
# splitted_dataset.data.mask = splitted_dataset.data.test_mask
# splitted_dataset.slices['mask'] = splitted_dataset.slices['train_mask']
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


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
# model = GM_GCN(n_layers = num_layers, input_dim = dim_node, hid_dim = dim_hidden, n_classes = num_classes)

# model = GM_GAT(num_layers, dim_node, 300, num_classes, heads = [7,4,1])
# model.to(device)
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GAT','GAT_100_best.pth')
# model.load_state_dict(torch.load(ckpt_path)['net'])

# model = GM_GCN(n_layers = num_layers, input_dim = dim_node, hid_dim = dim_hidden, n_classes = num_classes)
model = GM_GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
                shared_weights)
model.to(device)
# ckpt_path = osp.join('checkpoints', 'ba_community', 'GM_GCN','GM_GCN_100_best.pth')
# model.load_state_dict(torch.load(ckpt_path)['net'])

ckpt_path = osp.join('checkpoints', 'cora', 'GM_GCN2','GCN2_best.pth')
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
GraphMask = GraphMaskAdjMatProbe(vertex_dims, message_dims, num_classes, hidden_dims )
model.cuda()
GraphMask.cuda()
allowance =  0.2
pentalty_scalings = [5e-1]
entropy_scales = [1]
allowances = [0.1]
lr1s = [3e-2]
lr2s = [1e-3]
best_parameters = None
best_spar = 1
# for pentalty_scaling, entropy_scale, allowance,lr1, lr2 in tqdm.tqdm(product(pentalty_scalings, entropy_scales,allowances, lr1s, lr2s),
#                                                                      total = len(list(product(pentalty_scalings, entropy_scales,allowances, lr1s, lr2s)))):
for pentalty_scaling, entropy_scale, allowance,lr1, lr2 in product(pentalty_scalings, entropy_scales,allowances, lr1s, lr2s):
    explainer = GraphMaskExplainer(model, GraphMask, epoch = 1, penalty_scaling = pentalty_scaling,
                                   entropy_scale = entropy_scale,allowance = allowance, lr1 =lr1, lr2= lr2, batch_size = 50)

    import random
    import numpy as np
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)

    # state_dict = torch.load('gm_gcn100_bacom.pt')
    # GraphMask.load_state_dict(state_dict)
    explainer.train_graphmask(dataset)
    # probs, sizes = explainer.train_graphmask(dataset)
    # outputs = {'prob':np.array(probs), 'graph_size':np.array(sizes)}
    # seaborn.scatterplot(y = 'prob', x = 'graph_size', data = outputs)
    # seaborn.histplot(x = 'graph_size', data=outputs)
    # plt.show()
    # sys.exit()
    torch.save(GraphMask.state_dict(), 'checkpoints/explainer/cora/gm_gcn2_free_nobase.pt')
    #
    # state_dict = torch.load('checkpoints/explainer/cora/gm_gcn_free.pt')
    # GraphMask.load_state_dict(state_dict)
    # sys.exit()

    #
    #
    # node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    # from dig.xgraph.method.pgexplainer import PlotUtils
    # plotutils = PlotUtils(dataset_name='ba_community')
    # data = dataset[0].cuda()
    # dist_dict = {}
    # for i, (u, v) in enumerate(data.edge_index.t().tolist()):
    #     if u == 0 and v != 0:
    #         dist_dict[(u,v)] = 100
    #     elif u != 0 and v == 0:
    #         dist_dict[(u,v)] = 100
    #     elif u == 0 and v == 0:
    #         dist_dict[(u,v)] = 0.01
    #     else:
    #         dist_dict[(u,v)] = 0.25
    # node_idx = node_indices[0]
    #
    # new_data, subset, new_node_idx, mask= \
    #     explainer(data, node_idx=node_idx, y=data.y, top_k=6, visualize = True)
    #
    # visualize(new_data, edge_mask=mask, top_k=6, plot_utils=plotutils, node_idx= new_node_idx, vis_name = f'fig/graphmask_gat{node_idx}.pdf',dist_dict = dist_dict)
    # sys.exit()

    # --- Create data collector and explanation processor ---
    from dig.xgraph.evaluation import XCollector
    x_collector = XCollector()

    ## Run explainer on the given model and dataset
    a = 0.50
    b = 0.05
    c = []
    while a < 1:
        c.append(a)
        a += b
    # --- Set the Sparsity to 0.5
    fidelity1 = []
    fidelity2 = []
    inv_fidelity1 = []
    inv_fidelity2 = []
    in_motifs = []
    sparsitys = []
    data = dataset[0]
    subgraphs = {}
    explain_node_index_list = torch.where(data.test_mask)[0]
    for j, node_idx in tqdm.tqdm(enumerate(explain_node_index_list), total= len(explain_node_index_list)):
        x, edge_index, y, subset, _ = explainer.get_subgraph(node_idx, data.x, data.edge_index,data.y)
        subgraphs[j] = {'x':x.cpu(), 'edge_index':edge_index.cpu(), 'new_node_idx':torch.where(subset == node_idx)[0].cpu()}
    import pickle as pk
    for explanation_confidence in c:

        # large_index = pk.load(open('large_subgraph_bacom.pk','rb'))['node_idx']
        # motif = pk.load(open('Ba_Community_motif.plk','rb'))
        # explain_node_index_list = list(set(large_index).intersection(set(motif.keys())))

        spars = 0
        for j, node_idx in tqdm.tqdm(enumerate(explain_node_index_list), total= len(explain_node_index_list)):
        # for j, node_idx in enumerate(explain_node_index_list):
            data = data.to(device)
            subgraph = subgraphs[j]
            # if torch.isnan(data.y[0].squeeze()):
            #     continue
            with torch.no_grad():
                _, masks, related_preds = \
                    explainer(data, explanation_confidence = explanation_confidence, node_idx=node_idx, y=data.y,
                              x = subgraph['x'], edge_index = subgraph['edge_index'], new_node_idx = subgraph['new_node_idx'])
            spars += related_preds[0]['sparsity']

        # outputs = {'inv_fidelity': invs, 'sparsity': spars}
        # seaborn.scatterplot(y = 'inv_fidelity', x = 'sparsity', data = outputs)
        # plt.show()
        # outputs = {'graph_size': sizes, 'sparsity': spars}
        # seaborn.scatterplot(y = 'sparsity', x = 'graph_size', data = outputs)
        # plt.show()
        # sys.exit()
            # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
            # obtain the result: x_processor(data, masks, x_collector)
        #     if index >= 99:
        #         break
        #
        # if index >= 99:
        #     break
        sparsity = spars/(j + 1)
        sparsitys.append(sparsity)
        if sparsity < best_spar:
            best_parameters = [pentalty_scaling, entropy_scale, allowance,lr1, lr2]
            best_spar = sparsity
        print(f'parameters:{[pentalty_scaling, entropy_scale, allowance,lr1, lr2]}\n'
              f'explanation_confidence: {explanation_confidence:.2f}\n'
              f'Sparsity: {sparsity:.4f}')
    print('evaluation_confidence: ',c)
    print('sparsity: ', sparsitys)
print('best parameters: ', best_parameters,
     '\nbest inverse fidelity:', best_spar)