import sys
sys.path.append('../../..')
from tqdm import tqdm
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
dataset.data.x = dataset.data.x.to(torch.float32)
# dataset.data.x = dataset.data.x[:, :1]
# dataset.data.y = dataset.data.y[:, 2]
dim_node = dataset.num_node_features
dim_edge = dataset.num_edge_features
# num_targets = dataset.num_classes
num_classes = dataset.num_classes

splitted_dataset = split_dataset(dataset)
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

# model = GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
# shared_weights, dropout)
# model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
model = GM_GCN(num_layers, dim_node, dim_hidden, num_classes)
model.to(device)
check_checkpoints()
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GIN_2l', '0', 'GIN_2l_best.ckpt')
# model.load_state_dict(torch.load(ckpt_path)['state_dict'])
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN2','GCN2_best.pth')
# ckpt_path = osp.join('checkpoints', 'ba_shapes', 'GCN_2l','GCN_2l_best.pth')
ckpt_path = osp.join('checkpoints', 'ba_community', 'GM_GCN','GM_GCN_best.pth')
model.load_state_dict(torch.load(ckpt_path)['net'])
# from dig.xgraph.method import GNNExplainer

class Explainer(GNNExplainer):
    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, return_type: str = 'log_prob',
                 log: bool = True):
        super(GNNExplainer, self).__init__()
        assert return_type in ['log_prob', 'prob', 'raw']
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.return_type = return_type
        self.log = log
    def get_subgraph(self, node_idx, x, edge_index,y,**kwargs):

        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, mapping, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, self.num_hops, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        # mapping = {int(v): k for k, v in enumerate(subset)}
        # subgraph = graph.subgraph(subset.tolist())
        # nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        if y is not None:
            y = y[subset]
        return subset, x, edge_index,  mapping, edge_mask,y

    def visualize_subgraph(self, x,node_idx, edge_index, edge_mask, y=None,
                           threshold=None, **kwargs):
        r"""Visualizes the subgraph given an edge mask
        :attr:`edge_mask`.
        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`-1` to explain graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.
        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        import matplotlib.pyplot as plt
        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None

        else:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()
        subset, x, edge_index, mapping, hard_edge_mask,y = self.get_subgraph(node_idx, x, edge_index,y)
        data = Data(edge_index=edge_index, att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)

        node_kwargs = copy(kwargs)
        node_kwargs['node_size'] = kwargs.get('node_size') or 80
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        label_kwargs = copy(kwargs)
        label_kwargs['font_size'] = kwargs.get('font_size') or 5

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))
        plt.figure(1,figsize=(1000,1000), dpi = 2400)
        nx.draw_networkx_nodes(G, pos, node_color=y.tolist(), **node_kwargs)
        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G

    def explain_node(self, node_idx, x, edge_index, y,topk = 0, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)


        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(x=x, edge_index=edge_index, **kwargs)
            log_logits = self.__to_log_prob__(out)
            probs = F.softmax(out, dim = -1)
            pred_label = log_logits.argmax(dim=-1)
            label = pred_label[node_idx]

        # Only operate on a k-hop subgraph around `node_idx`.
        if topk > 0:
            subset, x, edge_index, mapping, hard_edge_mask,y = self.get_subgraph(
                node_idx, x, edge_index,y, **kwargs)
            new_node_idx = torch.where(subset == node_idx)[0]
        else:
            x, edge_index, mapping, hard_edge_mask, kwargs = self.__subgraph__(
                node_idx, x, edge_index, **kwargs)
        self.__set_masks__(x, edge_index)
        self.to(x.device)
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x
            out = self.model(x=h, edge_index=edge_index, **kwargs)
            log_logits = self.__to_log_prob__(out)
            loss = self.__loss__(mapping, log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()
        if topk > 0:
            data = Data(x=x, edge_index=edge_index,  y = y)
            selected_nodes = calculate_selected_nodes(data, edge_mask, topk)
            maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            value_func = GnnNets_NC2value_func(self.model,
                                               node_idx=new_node_idx,
                                               target_class=label)
            maskout_pred = gnn_score(maskout_nodes_list, data, value_func,
                                     subgraph_building_method='zero_filling')
            sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]
            related_preds = [{
                'maskout': maskout_pred,
                'origin': probs[label],
                'sparsity': sparsity_score}]
            return node_feat_mask, edge_mask, related_preds
        self.__clear_masks__()

        return node_feat_mask, edge_mask
explainer = Explainer(model, epochs=100, lr=0.01)
explainer.coeffs['edge_size'] = 1e-6
node_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
node_idx = node_indices[6]

data = dataset[0].cuda()
_, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index, data.y)
print(edge_mask)
thres_index = max(edge_mask.shape[0] - 12, 0)
threshold = float(edge_mask.reshape(-1).sort().values[thres_index])
hard_edge_mask = (edge_mask >= threshold)
print(torch.count_nonzero(hard_edge_mask))
ax, G = explainer.visualize_subgraph(data.x, node_idx, data.edge_index, hard_edge_mask,y=data.y)

# plt.show()
plt.savefig('gnn_explainer.png')
print(1)
# --- Set the Sparsity to 0.5 ---
sparsity = 0.5

# --- Create data collector and explanation processor ---
from dig.xgraph.evaluation import XCollector, ExplanationProcessor
x_collector = XCollector(sparsity)
x_processor = ExplanationProcessor(model=model, device=device)

index = -1
for i, data in enumerate(dataloader):
    for j, node_idx in enumerate(torch.where(data.train_mask == True)[0].tolist()):
        index += 1
        print(f'explain graph {i} node {node_idx}')
        data.to(device)

        if torch.isnan(data.y[0].squeeze()):
            continue

        _, masks, related_preds = \
            explainer.explain_node(node_idx, data.x, data.edge_index,data.y, topk = 6)

        x_collector.collect_data(masks, related_preds, data.y[node_idx])

        # if you only have the edge masks without related_pred, please feed sparsity controlled mask to
        # obtain the result: x_processor(data, masks, x_collector)
        if index >= 99:
            break

    if index >= 99:
        break

print(f'Fidelity: {x_collector.fidelity:.4f}\n'
      f'Fidelity_inv: {x_collector.fidelity_inv:.4f}\n'
      f'Sparsity: {x_collector.sparsity:.4f}')
