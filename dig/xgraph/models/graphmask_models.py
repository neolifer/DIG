import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_geometric.nn as gnn
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data.batch import Batch
import math
from typing import Callable, Union, Tuple, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor


class GM_GCNconv(gnn.GCNConv):
    def __init__(self, *args, **kwargs):
        super(GM_GCNconv, self).__init__(*args, **kwargs)

    def forward(self, x, edge_index,
                edge_weight: OptTensor = None, message_scale: Tensor = None,
                message_replacement: Tensor = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        self.last_edge_index = edge_index
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
        # print(original_message.shape)
        self.message_dim = original_message.shape[-1]
        if message_scale is not None:
            original_message = torch.mul(original_message, message_scale.unsqueeze(-1))
            if message_replacement is not None:
                message = original_message + torch.mul( message_replacement, (1 - message_scale).unsqueeze(-1))
        self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1) if message_scale is not None else torch.cat([x_j, x_i, original_message], dim = -1)
        # print(self.latest_vertex_embeddings.shape[0])
        if message_scale is not None:
            return message
        return original_message

    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        return self.message_dim

class GM_GCN(nn.Module):
    def __init__(self, n_layers, input_dim, hid_dim, n_classes, dropout = 0):
        super(GM_GCN, self).__init__()
        self.convs = nn.ModuleList([GM_GCNconv(input_dim, hid_dim)]
                                   + [
                                        GM_GCNconv(hid_dim, hid_dim)
                                        for _ in range(n_layers - 1)
                                   ]
                                   )
        self.hidden_dims = [hid_dim for _ in range(n_layers)]
        self.outlayer = nn.Linear(hid_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # self.bn = nn.BatchNorm1d(hid_dim)
        for conv in self.convs:
            conv.chache = None


    def forward(self, x, edge_index, message_scales = None, message_replacement = None,**kwargs):
        # x, edge_index = data.x, data.edge_index
        if message_scales and message_replacement:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index,message_scale=message_scales[i], message_replacement=message_replacement[i])
                # x = self.bn(x)
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

    def get_emb(self,*args):
        if len(args) == 1:
            x, edge_index = args[0].x, args[0].edge_index
        else:
            x, edge_index = args[0], args[1]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.relu(x)
        return x
    def get_latest_vertex_embedding(self):
        self.latest_vertex_embeddings = []
        for conv in self.convs:
            self.latest_vertex_embeddings.append(conv.get_latest_vertex_embedding())
            conv.latest_vertex_embeddings = None
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        self.message_dims = []
        for conv in self.convs:
            self.message_dims.append(conv.get_message_dim())
        return self.message_dims

    def get_last_edge_index(self):
        self.last_edge_indexs = []
        for conv in self.convs:
            self.last_edge_indexs.append(conv.last_edge_index)
        return self.last_edge_indexs


class GM_GCN2Conv(gnn.GCN2Conv):
    def __innit__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_weights = None
    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, message_scale = None, message_replacement = None) -> Tensor:
        """"""
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None, message_scale=message_scale, message_replacement=message_replacement)
        if x_0 is None:
            return x
        x.mul_(1 - self.alpha)

        x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
            out += torch.addmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                               alpha=self.beta)
        self.edge_weights = edge_weight

        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_weight: OptTensor, message_scale: Tensor,
                message_replacement: Tensor) -> Tensor:
        original_message = x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        # print(original_message.shape)
        self.message_dim = original_message.shape[-1]
        if message_scale is not None:
            original_message = torch.mul(original_message, message_scale.unsqueeze(-1))
            if message_replacement is not None:
                message = original_message + torch.mul( message_replacement, (1 - message_scale).unsqueeze(-1))
        self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1) if message_scale is not None else torch.cat([x_j, x_i, original_message], dim = -1)
        # print(self.latest_vertex_embeddings.shape[0])
        if message_scale is not None:
            return message
        return original_message

    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        return self.message_dim


class GM_GCN2(nn.Module):
    def __init__(self, model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers, shared_weights,dropout):
        super().__init__()

        convs = []
        for i in range( num_layers ):
            convs.append(GM_GCN2Conv(dim_hidden, alpha, theta, i + 1))
        self.convs = nn.ModuleList(
            convs
        )
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(dim_node, dim_hidden))
        self.fcs.append(nn.Linear(dim_hidden, num_classes))
        self.relu = nn.ReLU()
        self.dropout = dropout




    def forward(self, x, edge_index, message_scales = None, message_replacement = None):
        """
        :param Required[data]: Batch - input data
        :return:
        """
        # x, edge_index = data.x, data.edge_index
        if message_scales is not None and message_replacement is not None:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(self.fcs[0](x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(self.convs[0](x, None, edge_index, message_scale = message_scales[0], message_replacement = message_replacement[0]))
            x_0 = x
            for i, conv in enumerate(self.convs):
                if i == 0:
                    continue
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.relu(conv(x, x_0, edge_index, message_scale = message_scales[i], message_replacement = message_replacement[i]))
            out = self.fcs[-1](x)

            return out
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.fcs[0](x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.relu(self.convs[0](x, None, edge_index))
        x_0 = x
        for i,conv in enumerate(self.convs):
            if i == 0:
                continue
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.relu(conv(x, x_0, edge_index))

        out = self.fcs[-1](x)

        return out

    def get_latest_vertex_embedding(self):
        self.latest_vertex_embeddings = []
        for conv in self.convs:
            self.latest_vertex_embeddings.append(conv.get_latest_vertex_embedding())
            conv.lates_vertex_embeddings = None
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        self.message_dims = []
        for conv in self.convs:
            self.message_dims.append(conv.get_message_dim())
        return self.message_dims

    def get_last_edge_index(self):
        self.last_edge_indexs = []
        for conv in self.convs:
            self.last_edge_indexs.append(conv.last_edge_index)
        return self.last_edge_indexs

class GM_GATConv(gnn.GATConv):
    def __init__(self,*args, **kwargs):
        super(GM_GATConv, self).__init__(*args, **kwargs)
        self.get_vertex = True
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, message_scale = None, message_replacement = None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        self.last_edge_index = edge_index
        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size, message_scale = message_scale, message_replacement = message_replacement)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i:Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], message_scale: Tensor,
                message_replacement: Tensor) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        original_message =  x_j * alpha.unsqueeze(-1)
        self.message_dim = original_message.shape[-1]
        if message_scale is not None:
            original_message = torch.mul(original_message, message_scale.unsqueeze(-1))
            if message_replacement is not None:
                message = original_message + torch.mul( message_replacement, (1 - message_scale).unsqueeze(-1))
        if self.get_vertex:
            self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1) if message_scale is not None else torch.cat([x_j, x_i, original_message], dim = -1)
        # print(self.latest_vertex_embeddings.shape[0])
        if message_scale is not None:
            return message
        return original_message

    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        return self.message_dim

class GM_GAT(nn.Module):
    def __init__(self,n_layers, input_dim, hid_dim, n_classes, dropout = 0, heads = None ):
        super(GM_GAT, self).__init__()

        if not heads:
            heads = [3 for _ in range(n_layers)]
        self.convs = nn.ModuleList([GM_GATConv(input_dim, hid_dim, heads = heads[0], dropout = dropout)]
                                   + [
                                       GM_GATConv(heads[i]*hid_dim, hid_dim, heads = heads[i + 1], dropout = dropout)
                                       for i in range(n_layers - 1)
                                   ]
                                   )
        self.hidden_dims = [hid_dim for _ in range(n_layers)]
        self.outlayer = nn.Linear(heads[-1]*hid_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hid_dim)

    def set_get_vertex(self, get_vertex = True):
        for conv in self.convs:
            conv.get_vertex = get_vertex
    def forward(self, x, edge_index, message_scales = None, message_replacement = None):
        # x, edge_index = data.x, data.edge_index
        if message_scales and message_replacement:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index,message_scale=message_scales[i], message_replacement=message_replacement[i])
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
            self.latest_vertex_embeddings.append(conv.get_latest_vertex_embedding())
            conv.latest_vertex_embeddings = None
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        self.message_dims = []
        for conv in self.convs:
            self.message_dims.append(conv.get_message_dim())
        return self.message_dims

    def get_last_edge_index(self):
        self.last_edge_indexs = []
        for conv in self.convs:
            self.last_edge_indexs.append(conv.last_edge_index)
        return self.last_edge_indexs


class GM_SAGEConv(gnn.SAGEConv):
    def __init__(self,*args, **kwargs):
        super(GM_SAGEConv, self).__init__()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, message_scale: Tensor = None,
                message_replacement: Tensor = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size, message_scale = message_scale, message_replacement = message_replacement)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, x_i: Tensor,  message_scale: Tensor = None,
                message_replacement: Tensor = None) -> Tensor:
        original_message =  x_j
        self.message_dim = original_message.shape[-1]
        if message_scale is not None:
            original_message = torch.mul(original_message, message_scale.unsqueeze(-1))
            if message_replacement is not None:
                message = original_message + torch.mul( message_replacement, (1 - message_scale).unsqueeze(-1))
        self.latest_vertex_embeddings = torch.cat([x_j, x_i, message], dim = -1) if message_scale is not None else torch.cat([x_j, x_i, original_message], dim = -1)
        # print(self.latest_vertex_embeddings.shape[0])
        if message_scale is not None:
            return message
        return original_message

    def get_latest_vertex_embedding(self):
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        return self.message_dim

class GM_SAGE(nn.Module):
    def __init__(self, n_layers, input_dim, hid_dim, n_classes, dropout = 0):
        super(GM_SAGE, self).__init__()
        self.convs = nn.ModuleList([GM_SAGEConv(input_dim, hid_dim)]
                                   + [
                                       GM_SAGEConv(hid_dim, hid_dim)
                                       for _ in range(n_layers - 1)
                                   ]
                                   )
        self.hidden_dims = [hid_dim for _ in range(n_layers)]
        self.outlayer = nn.Linear(hid_dim, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, message_scales = None, message_replacement = None):
        # x, edge_index = data.x, data.edge_index
        if message_scales and message_replacement:
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index,message_scale=message_scales[i], message_replacement=message_replacement[i])
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
            self.latest_vertex_embeddings.append(conv.get_latest_vertex_embedding())
        return self.latest_vertex_embeddings

    def get_message_dim(self):
        self.message_dims = []
        for conv in self.convs:
            self.message_dims.append(conv.get_message_dim())
        return self.message_dims

    def get_last_edge_index(self):
        self.last_edge_indexs = []
        for conv in self.convs:
            self.last_edge_indexs.append(conv.last_edge_index)
        return self.last_edge_indexs
