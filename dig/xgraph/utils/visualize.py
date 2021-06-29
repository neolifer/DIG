from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_networkx
import networkx as nx
from typing import Tuple, List, Dict, Optional
from ..method.pgexplainer import PlotUtils
from torch import Tensor
import torch


def visualize(data: Data, edge_mask: Tensor, top_k: int, plot_utils: PlotUtils,
              words: Optional[list] = None, node_idx: int = None, model = None,  vis_name: Optional[str] = None, vis_graph = False ):
    if vis_name is None:
        vis_name = f"filename.png"

    data = data.to('cpu')
    edge_mask = edge_mask.to('cpu')
    if vis_graph:
        pass
    else:
        assert node_idx is not None, "visualization method doesn't get the target node index"
        new_node_idx = node_idx
        new_data = data
        graph = to_networkx(new_data)
        y = data.y
        plot_utils.plot_soft_edge_mask(graph,
                                       edge_mask,
                                       top_k=top_k,
                                       un_directed=True,
                                       y=y,
                                       node_idx=new_node_idx,
                                       figname=vis_name)
