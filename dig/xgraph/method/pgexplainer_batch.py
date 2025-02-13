"""
Description: The implement of PGExplainer model
<https://arxiv.org/abs/2011.04573>
"""
import sys
from typing import Optional
from math import sqrt

import time
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.data import Data, Batch,DataLoader
import tqdm
import networkx as nx
from textwrap import wrap
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_networkx
from torch_geometric.utils.num_nodes import maybe_num_nodes
from typing import Tuple, List, Dict, Optional
from .shapley import GnnNets_GC2value_func, GnnNets_NC2value_func, gnn_score
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem

EPS = 1e-6


class Temp_data(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, node_idx = None, pred = None, emb = None,**kwargs):
        super(Temp_data, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.pred = pred
        self.emb = emb
    # def __inc__(self, key, value):
    #     if key == 'node_idx':
    #         return 1
    #     elif key == 'pred':
    #         return self.pred.size(0)
    #     elif key == 'emb':
    #         return self.emb.size(0)
    #     else:
    #         return super(Temp_data, self).__inc__(key, value)


def k_hop_subgraph_with_default_whole_graph(
        edge_index, node_idx=None, num_hops=5, relabel_nodes=False,
        num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']

    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        if num_hops > 3:
            num_hops = 3
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx


def calculate_selected_nodes(data, edge_mask, top_k):
    threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
    hard_mask = (edge_mask > threshold).cpu()
    edge_idx_list = torch.where(hard_mask == 1)[0]
    selected_nodes = []
    edge_index = data.edge_index.cpu().numpy()
    for edge_idx in edge_idx_list:
        selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
    selected_nodes = list(set(selected_nodes))
    return selected_nodes


class PlotUtils(object):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def plot_subgraph(self, graph, nodelist, colors='#FFA500', labels=None, edge_color='gray',
                      edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                        n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        # plt.show()
        plt.close('all')

    def plot_subgraph_with_nodes(self, graph, nodelist, node_idx, colors='#FFA500', labels=None, edge_color='gray',
                                 edgelist=None, subgraph_edge_color='black', title_sentence=None, figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges() if
                        n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph) # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=60)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=200)

        nx.draw_networkx_edges(graph, pos, width=1, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=2,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        plt.figure(1,figsize=(18,18), dpi = 100)
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))
        # plt.show()
        plt.savefig(figname)

    def plot_ba2motifs(self, graph, nodelist, edgelist=None, figname=None):
        return self.plot_subgraph(graph, nodelist, edgelist=edgelist, figname=figname)

    def plot_molecule(self, graph, nodelist, x, edgelist=None, figname=None):
        # collect the text information and node color
        if self.dataset_name == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in MoleculeNet.names.keys():
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist, colors=colors, labels=node_labels,
                           edgelist=edgelist, edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=None, figname=figname)

    def plot_sentence(self, graph, nodelist, words, edgelist=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]
            nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=5,
                                   edge_color='yellow', arrows=False)

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='grey', arrows=False)
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if figname is not None:
            plt.savefig(figname)
        plt.close('all')

    def plot_bashapes(self, graph, nodelist, y, node_idx, edgelist=None, figname=None):

        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph, nodelist, node_idx, colors, edgelist=edgelist, figname=figname,
                                      subgraph_edge_color='black')

    def plot_bacom(self, graph, nodelist, y, node_idx, edgelist=None, figname=None):

        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#1B4F72', '#85C1E9', 'green', '#E74C3C', '#D2B4DE','#641E16', '#154360']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph, nodelist, node_idx, colors, edgelist=edgelist, figname=figname,
                                      subgraph_edge_color='black')

    def get_topk_edges_subgraph(self, edge_index, edge_mask, top_k, un_directed=False):
        if un_directed:
            top_k = 2 * top_k
        edge_mask = edge_mask.reshape(-1)
        print(edge_mask)
        thres_index = max(edge_mask.shape[0] - top_k, 0)
        threshold = float(edge_mask.reshape(-1).sort().values[thres_index])
        hard_edge_mask = (edge_mask >= threshold)
        # print(torch.sum(hard_edge_mask))

        selected_edge_idx = np.where(hard_edge_mask == 1)[0].tolist()
        nodelist = []
        edgelist = []
        for edge_idx in selected_edge_idx:
            edges = edge_index[:, edge_idx].tolist()
            nodelist += [int(edges[0]), int(edges[1])]
            edgelist.append((edges[0], edges[1]))
        nodelist = list(set(nodelist))
        return nodelist, edgelist

    def plot_soft_edge_mask(self, graph, edge_mask, top_k, un_directed, figname, **kwargs):
        edge_index = torch.tensor(list(graph.edges())).T
        try:
            edge_mask = edge_mask.clone().detach()
        except:
            pass
        if self.dataset_name.lower() in ['ba_2motifs', 'ba_lrp']:
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_ba2motifs(graph, nodelist, edgelist, figname=figname)

        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_molecule(graph, nodelist, x, edgelist, figname=figname)

        elif self.dataset_name.lower() in ['ba_shapes', 'tree_grid', 'tree_cycle']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_bashapes(graph, nodelist, y, node_idx, edgelist, figname=figname)

        elif self.dataset_name.lower() in ['ba_community']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_bacom(graph, nodelist, y, node_idx, edgelist, figname=figname)

        elif self.dataset_name.lower() in ['Graph_SST2'.lower()]:
            words = kwargs.get('words')
            nodelist, edgelist = self.get_topk_edges_subgraph(edge_index, edge_mask, top_k, un_directed)
            self.plot_sentence(graph, nodelist, words=words, edgelist=edgelist, figname=figname)

        else:
            raise NotImplementedError


class PGExplainer(nn.Module):
    r"""
    An implementation of PGExplainer in
    `Parameterized Explainer for Graph Neural Network <https://arxiv.org/abs/2011.04573>`_.

    Args:
        model (:class:`torch.nn.Module`): The target model prepared to explain
        in_channels (:obj:`int`): Number of input channels for the explanation network
        explain_graph (:obj:`bool`): Whether to explain graph classification model (default: :obj:`True`)
        epochs (:obj:`int`): Number of epochs to train the explanation network
        lr (:obj:`float`): Learning rate to train the explanation network
        coff_size (:obj:`float`): Size regularization to constrain the explanation size
        coff_ent (:obj:`float`): Entropy regularization to constrain the connectivity of explanation
        t0 (:obj:`float`): The temperature at the first epoch
        t1(:obj:`float`): The temperature at the final epoch
        num_hops (:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
        (default: :obj:`None`)

    .. note: For node classification model, the :attr:`explain_graph` flag is False.
      If :attr:`num_hops` is set to :obj:`None`, it will be automatically calculated by calculating the
      :class:`torch_geometric.nn.MessagePassing` layers in the :attr:`model`.

    """
    def __init__(self, model, in_channels: int, device, explain_graph: bool = False, epochs: int = 20,
                 lr: float = 0.001, coff_size: float = 0.001, coff_ent: float = 0.01, coff_pred: float = 1.0,
                 t0: float = 5.0, t1: float = 1.0, batch_size: int = 1, num_hops: Optional[int] = None):
        super(PGExplainer, self).__init__()
        self.model = model
        self.device = 'cuda:0'
        self.model.to(self.device)
        self.in_channels = in_channels
        self.explain_graph = explain_graph
        self.batch_size = batch_size
        # training parameters for PGExplainer
        self.epochs = epochs
        self.lr = lr
        self.coff_size = coff_size
        self.coff_ent = coff_ent
        self.t0 = t0
        self.t1 = t1

        self.num_hops = self.update_num_hops(num_hops)
        self.init_bias = 0.0

        # Explanation model in PGExplainer
        self.elayers = nn.ModuleList()
        self.elayers.append(nn.Sequential(nn.Linear(in_channels, 64), nn.ReLU()))
        self.elayers.append(nn.Linear(64, 1))
        self.elayers.to(self.device)

    def __set_masks__(self, x: Tensor, edge_index: Tensor, edge_mask: Tensor = None):
        r""" Set the edge weights before message passing

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            edge_mask (:obj:`torch.Tensor`): Edge weight matrix before message passing
              (default: :obj:`None`)

        The :attr:`edge_mask` will be randomly initialized when set to :obj:`None`.

        .. note:: When you use the :meth:`~PGExplainer.__set_masks__`,
          the explain flag for all the :class:`torch_geometric.nn.MessagePassing`
          modules in :attr:`model` will be assigned with :obj:`True`. In addition,
          the :attr:`edge_mask` will be assigned to all the modules.
          Please take :meth:`~PGExplainer.__clear_masks__` to reset.
        """
        (N, F), E = x.size(), edge_index.size(1)
        std = 0.1
        init_bias = self.init_bias
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * std)
        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))

        if edge_mask is None:
            self.edge_mask = torch.nn.Parameter(torch.randn(E) * std + init_bias)
        else:
            self.edge_mask = edge_mask

        self.edge_mask.to(self.device)
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        """ clear the edge weights to None, and set the explain flag to :obj:`False` """
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    def update_num_hops(self, num_hops: int):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __loss__(self, prob: Tensor, ori_pred: int, mask_sigmoid):
        # logit = prob[ori_pred]
        # logit = logit + EPS
        # pred_loss = - torch.log(logit)
        loss = nn.CrossEntropyLoss()

        pred_loss = loss(prob, ori_pred)

        # size
        edge_mask = mask_sigmoid
        size_loss = self.coff_size * torch.sum(edge_mask)

        # entropy

        edge_mask = edge_mask * 0.99 + 0.005
        mask_ent = - edge_mask * torch.log(edge_mask) - (1 - edge_mask) * torch.log(1 - edge_mask)

        mask_ent_loss = self.coff_ent * torch.mean(mask_ent)

        loss = pred_loss + size_loss + mask_ent_loss
        return loss

    def get_subgraph(self,
                     node_idx: int,
                     x: Tensor,
                     edge_index: Tensor,
                     y: Optional[Tensor] = None,
                     **kwargs) \
            -> Tuple[Tensor, Tensor, Tensor, List, Dict]:
        r""" extract the subgraph of target node

        Args:
            node_idx (:obj:`int`): The node index
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            y (:obj:`torch.Tensor`, :obj:`None`): Node label matrix with shape :obj:`[num_nodes]`
              (default :obj:`None`)
            kwargs(:obj:`Dict`, :obj:`None`): Additional parameters

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`, :class:`torch.Tensor`,
          :obj:`List`, :class:`Dict`)

        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, self.num_hops, relabel_nodes=True,
            num_nodes=num_nodes, flow=self.__flow__())

        mapping = {int(v): k for k, v in enumerate(subset)}
        subgraph = graph.subgraph(subset.tolist())
        nx.relabel_nodes(subgraph, mapping)

        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item
        if y is not None:
            y = y[subset]
        return x, edge_index, y, subset, kwargs

    def concrete_sample(self, log_alpha: Tensor, beta: float = 1.0, training: bool = True):
        r""" Sample from the instantiation of concrete distribution when training """
        if training:
            random_noise = torch.rand(log_alpha.shape)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha

        return gate_inputs

    def explain(self,
                x, edge_index,
                embed: Tensor,
                tmp: float = 1.0,
                training: bool = False,
                node_idx: int = None,
                emb_self: Tensor = None) \
            -> Tuple[float, Tensor]:
        r""" explain the GNN behavior for graph with explanation network

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape

              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            embed (:obj:`torch.Tensor`): Node embedding matrix with shape :obj:`[num_nodes, dim_embedding]`
            tmp (:obj`float`): The temperature parameter fed to the sample procedure
            training (:obj:`bool`): Whether in training procedure or not
            node_index(:obj: 'int'): Node index for node classification
        Returns:
            probs (:obj:`torch.Tensor`): The classification probability for graph with edge mask
            edge_mask (:obj:`torch.Tensor`): The probability mask for graph edges
        """
        nodesize = embed.shape[0]
        print(nodesize)
        feature_dim = embed.shape[-1]
        if not training:
            if self.explain_graph:
                col, row = edge_index
                f1 = embed[col]
                f2 = embed[row]
                f12self = torch.cat([f1, f2], dim=-1)
            else:
                col, row = edge_index
                f1 = embed[col]
                f2 = embed[row]
                self_embed = embed[node_idx].repeat(f1.shape[0], 1)
                f12self = torch.cat([f1, f2, self_embed], dim=-1)
        else:
            f12self = embed
        print(torch.cuda.memory_reserved(0)/1073741824)
        h = f12self.to(self.device)

        for elayer in self.elayers:
            h = elayer(h)
        values = h.reshape(-1)
        print(torch.cuda.memory_reserved(0)/1073741824)
        values = self.concrete_sample(values, beta=tmp, training=training)
        print(torch.cuda.memory_reserved(0)/1073741824)
        mask_sparse = torch.sparse_coo_tensor(
            edge_index, values, (nodesize, nodesize)
        )
        print(torch.cuda.memory_reserved(0)/1073741824)
        mask_sigmoid = mask_sparse.to_dense()
        print(torch.cuda.memory_reserved(0)/1073741824)
        # set the symmetric edge weights
        sym_mask = (mask_sigmoid + mask_sigmoid.transpose(0, 1)) / 2
        edge_mask = sym_mask[edge_index[0], edge_index[1]]

        # inverse the weights before sigmoid in MessagePassing Module
        # edge_mask = inv_sigmoid(edge_mask)
        self.__clear_masks__()
        self.__set_masks__(x, edge_index, edge_mask)

        # the model prediction with edge mask
        logits = self.model(x, edge_index)
        probs = logits

        self.__clear_masks__()
        return probs, edge_mask, mask_sigmoid

    def train_explanation_network(self, dataset, batch):
        r""" training the explanation network by gradient descent(GD) using Adam optimizer """
        optimizer = Adam(self.elayers.parameters(), lr=self.lr, weight_decay=1e-5)
        if self.explain_graph:
            with torch.no_grad():
                dataset_indices = list(range(len(dataset)))
                self.model.eval()
                emb_dict = {}
                ori_pred_dict = {}
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid].to(self.device)
                    logits = self.model(data)
                    emb = self.model.get_emb(data)
                    emb_dict[gid] = emb.data.cpu()
                    ori_pred_dict[gid] = logits.argmax(-1).data.cpu()

            # train the mask generator
            duration = 0.0
            for epoch in range(10):
                loss = 0.0
                pred_list = []
                # tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                tmp = 1
                self.elayers.train()
                optimizer.zero_grad()
                tic = time.perf_counter()
                for gid in tqdm.tqdm(dataset_indices):
                    data = dataset[gid]
                    data.to(self.device)
                    prob, _, mask_sigmoid = self.explain(data, embed=emb_dict[gid], tmp=tmp, training=True)
                    loss_tmp = self.__loss__(prob.squeeze(), ori_pred_dict[gid], mask_sigmoid)
                    loss_tmp.backward()
                    loss += loss_tmp.item()
                    pred_label = prob.argmax(-1).item()
                    pred_list.append(pred_label)

                optimizer.step()
                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss}')
        else:
            with torch.no_grad():
                data = dataset[0]
                data.to(self.device)
                self.model.eval()
                explain_node_index_list = list(range(data.y.shape[-1]))
                datalist = []
                for node_idx in tqdm.tqdm(explain_node_index_list):
                    x, edge_index, y, subset, _ = \
                        self.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
                    emb = self.model.get_emb(x, edge_index).detach()
                    col, row = edge_index
                    f1 = emb[col]
                    f2 = emb[row]
                    new_nodeidx = int(torch.where(subset == node_idx)[0])
                    self_embed = emb[new_nodeidx].repeat(f1.shape[0], 1)
                    f12self = torch.cat([f1, f2, self_embed], dim=-1)

                    datalist.append(Temp_data(x = x.detach().cpu(), edge_index = edge_index.detach().cpu(), node_idx = new_nodeidx, emb = f12self.detach().cpu()))
            loader = DataLoader(datalist, batch_size=self.batch_size, shuffle=False)
            # train the mask generator
            duration = 0.0
            for epoch in range(self.epochs):
                loss = 0.0
                optimizer.zero_grad()
                # tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
                tmp = 1
                self.elayers.train()
                tic = time.perf_counter()
                self.model.eval()
                for Batch in tqdm.tqdm(loader):
                    # r = torch.cuda.memory_reserved(0)/1073741824
                    # print(r)
                    optimizer.zero_grad()
                    x, edge_index, new_node_index, emb= Batch.x, Batch.edge_index, Batch.node_idx, Batch.emb
                    print(x.shape)
                    sys.exit()
                    x = x.to('cuda:0')
                    edge_index = edge_index.to('cuda:0')
                    emb = emb.to('cuda:0')
                    print(torch.cuda.memory_reserved(0)/1073741824)
                    pred, _, mask_sigmoid = self.explain(x, edge_index, emb, tmp, training=True, node_idx=new_node_index)
                    real_pred = self.model(x, edge_index)
                    real_pred = real_pred.argmax(-1).detach()
                    loss_tmp = self.__loss__(pred[new_node_index], real_pred[new_node_index], mask_sigmoid)
                    loss_tmp.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    # torch.cuda.empty_cache()
                    loss += loss_tmp.detach().item()

                duration += time.perf_counter() - tic
                print(f'Epoch: {epoch} | Loss: {loss}')

            print(f"training time is {duration:.5}s")

    def forward(self,
                data,
                **kwargs) \
            -> Tuple[None, List, List[Dict]]:
        r""" explain the GNN behavior for graph and calculate the metric values.
        The interface for the :class:`dig.evaluation.XCollector`.

        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            kwargs(:obj:`Dict`):
              The additional parameters
                - top_k (:obj:`int`): The number of edges in the final explanation results
                - y (:obj:`torch.Tensor`): The ground-truth labels

        :rtype: (:obj:`None`, List[torch.Tensor], List[Dict])
        """
        # set default subgraph with 10 edges
        top_k = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10

        y = kwargs.get('y')
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = y.to(self.device)

        self.__clear_masks__()
        logits = self.model(data)
        probs = F.softmax(logits, dim=-1)
        embed = self.model.get_emb(data)

        if self.explain_graph:
            # original value
            probs = probs.squeeze()
            label = y
            # masked value
            _, edge_mask, _ = self.explain(data, embed=embed, tmp=1.0, training=False)
            data = Data(x=x, edge_index=edge_index)
            selected_nodes = calculate_selected_nodes(data, edge_mask, top_k)
            maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            value_func = GnnNets_GC2value_func(self.model, target_class=label)
            maskout_pred = gnn_score(maskout_nodes_list, data, value_func,
                                     subgraph_building_method='zero_filling')
            sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]
        else:
            node_idx = kwargs.get('node_idx')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            # original value
            probs = probs.squeeze()[node_idx]
            label = y[node_idx]
            # masked value
            x, edge_index, _, subset, _ = self.get_subgraph(node_idx, x, edge_index)
            new_node_idx = torch.where(subset == node_idx)[0]
            data = Data(x, edge_index)
            embed = self.model.get_emb(data)

            _, edge_mask, _ = self.explain(x, edge_index, embed, tmp=1.0, training=False, node_idx=int(new_node_idx))

            selected_nodes = calculate_selected_nodes(data, edge_mask, top_k)
            maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
            value_func = GnnNets_NC2value_func(self.model,
                                               node_idx=new_node_idx,
                                               target_class=label)
            maskout_pred = gnn_score(maskout_nodes_list, data, value_func,
                                     subgraph_building_method='zero_filling')
            sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]

        # return variables
        pred_mask = [edge_mask.detach()]
        related_preds = [{
            'maskout': maskout_pred,
            'origin': probs[label],
            'sparsity': sparsity_score}]
        return None, pred_mask, related_preds

    def visualization(self, data: Data, edge_mask: Tensor, top_k: int, plot_utils: PlotUtils,
                      words: Optional[list] = None, node_idx: int = None, vis_name: Optional[str] = None):
        if vis_name is None:
            vis_name = f"filename.png"

        data = data.to('cpu')
        edge_mask = edge_mask.to('cpu')
        if self.explain_graph:
            graph = to_networkx(data)
            if words is None:
                plot_utils.plot_soft_edge_mask(graph,
                                               edge_mask,
                                               top_k=top_k,
                                               un_directed=True,
                                               words=words,
                                               figname=vis_name)
            else:
                plot_utils.plot_soft_edge_mask(graph,
                                               edge_mask,
                                               top_k=top_k,
                                               un_directed=True,
                                               x=x,
                                               figname=vis_name)
        else:
            assert node_idx is not None, "visualization method doesn't get the target node index"
            x, edge_index, y, subset, kwargs = \
                self.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
            new_node_idx = torch.where(subset == node_idx)[0]
            new_data = Data(x=x, edge_index=edge_index)
            graph = to_networkx(new_data)
            plot_utils.plot_soft_edge_mask(graph,
                                           edge_mask,
                                           top_k=top_k,
                                           un_directed=True,
                                           y=y,
                                           node_idx=new_node_idx,
                                           figname=vis_name)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
