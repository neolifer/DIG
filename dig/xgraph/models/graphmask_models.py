import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as gnn
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data.batch import Batch

from typing import Callable, Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor

from torch_sparse import SparseTensor


class GM_GCNconv(gnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GM_GCNconv, self).__init__(*args, **kwargs)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, message_scale: float = None, message_replacement: Tensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = torch.matmul(x, self.weight)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None, message_scale=message_scale, message_replacement=message_replacement)

        if self.bias is not None:
            out += self.bias

        return out
