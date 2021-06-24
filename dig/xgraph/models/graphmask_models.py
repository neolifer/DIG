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

    def forward(self, data,
                edge_weight: OptTensor = None, message_scale: Tensor = None,
                message_replacement: Tensor = None) -> Tensor:
        """"""
        x, edge_index = data.x, data.edge_index
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

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: OptTensor, message_scale: Tensor,
                message_replacement: Tensor) -> Tensor:
        original_message = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        if message_scale:
            original_message = original_message * message_scale.unsqueeze(-1)
            if message_replacement:
                message = original_message + (1 - message_scale).unsqueeze(-1) * message_replacement
        self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1)
        return message

    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

class GM_GCN(nn.Module):
    def __init__(self, n_layers, input_dim, hid_dim, n_classes, dropout = 0):
        super(GM_GCN, self).__init__()
        self.convs = nn.ModuleList([GM_GCNconv(input_dim, hid_dim)]
                                   + [
                                        GM_GCNconv(hid_dim, hid_dim)
                                        for _ in range(n_layers - 1)
                                   ]
                                   )
        self.outlayer = nn.Linear(hid_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.latest_vertex_embeddings = []


    def forward(self, data, message_scales = None, message_replacement = None):
        x, edge_index = data.x, data.edge_index
        if message_scales and message_replacement:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index,message_scales[i], message_replacement[i])
                x = self.relu(x)
                x = self.dropout(x)
            x = self.outlayer(x)
            return x
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.outlayer(x)
        return x

    def get_latest_vertex_embedding(self):
        self.latest_vertex_embeddings = []
        for conv in self.convs:
            self.latest_vertex_embeddings.append(conv.get_latest_vertex_embedding)
        return self.latest_vertex_embeddings