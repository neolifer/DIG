import sys

from torch_geometric.data import Data, Batch

import networkx as nx
from typing import Tuple, List, Dict, Optional
from ..method.pgexplainer import PlotUtils
from torch import Tensor
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
def to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=False,
                remove_self_loops=False, dist_dict = None):
    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool, optional): If set to :obj:`True`, will return a
            a :obj:`networkx.Graph` instead of a :obj:`networkx.DiGraph`. The
            undirected graph will correspond to the upper triangle of the
            corresponding adjacency matrix. (default: :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)
    """

    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    G.add_nodes_from(torch.unique(data.edge_index[0]).tolist())

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected and v > u:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)
        for key in edge_attrs if edge_attrs is not None else []:
            G[u][v][key] = values[key][i]

    for key in node_attrs if node_attrs is not None else []:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    return G


def visualize(data: Data, edge_mask: Tensor, top_k: int, plot_utils: PlotUtils,
              words: Optional[list] = None, node_idx: int = None, model = None,  vis_name: Optional[str] = None, vis_graph = False, dist_dict = None ):
    if vis_name is None:
        vis_name = f"filename.pdf"

    data = data.to('cpu')
    try:
        edge_mask = edge_mask.to('cpu')
    except:
        pass
    if vis_graph:
        pass
    else:
        assert node_idx is not None, "visualization method doesn't get the target node index"
        new_node_idx = node_idx
        new_data = data
        graph = to_networkx(new_data, dist_dict = dist_dict)
        y = data.y

        plot_utils.plot_soft_edge_mask(graph,
                                       edge_mask,
                                       top_k=top_k,
                                       un_directed=True,
                                       y=y,
                                       node_idx=new_node_idx,
                                       figname=vis_name)
