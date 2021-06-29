"""
FileName: utils.py
Description: The utils we may use for GNN model or Explainable model construction
Time: 2020/7/31 11:29
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes





class ReadOut(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def divided_graph(x, batch_index):
        graph = []
        for i in range(batch_index[-1] + 1):
            graph.append(x[batch_index == i])

        return graph

    def forward(self, x: torch.tensor, batch_index) -> torch.tensor:
        graph = ReadOut.divided_graph(x, batch_index)

        for i in range(len(graph)):
            graph[i] = graph[i].mean(dim=0).unsqueeze(0)

        out_readoout = torch.cat(graph, dim=0)

        return out_readoout


def normalize(x: torch.Tensor):
    x -= x.min()
    if x.max() == 0:
        return torch.zeros(x.size(), device=x.device)
    x /= x.max()
    return x


def subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
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
        num_hops: (int): The number of hops :math:`k`. when num_hops == -1,
            the whole graph will be returned.
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
        col, row = edge_index # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
    else:
        node_idx = node_idx.to(row.device)


    inv = None

    if num_hops != -1:
        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]
    else:
        subsets = node_idx
        cur_subsets = node_idx
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



    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask


class LagrangianOptimization:

    min_alpha = None
    max_alpha = None
    device = None
    original_optimizer = None
    batch_size_multiplier = None
    update_counter = 0

    def __init__(self, original_optimizer, device, init_alpha=0.55, min_alpha=-2, max_alpha=30, alpha_optimizer_lr=1e-1, batch_size_multiplier=None):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.device = device
        self.batch_size_multiplier = batch_size_multiplier
        self.update_counter = 0

        self.alpha = torch.tensor(init_alpha, device=device, requires_grad=True)
        self.optimizer_alpha = torch.optim.RMSprop([self.alpha], lr=alpha_optimizer_lr, centered=True)
        self.original_optimizer = original_optimizer

    def update(self, f, g, model):
        """
        L(x, lambda) = f(x) + lambda g(x)

        :param f_function:
        :param g_function:
        :return:
        """

        # if self.batch_size_multiplier is not None and self.batch_size_multiplier > 1:
        #     if self.update_counter % self.batch_size_multiplier == 0:
        #         self.original_optimizer.zero_grad()
        #         self.optimizer_alpha.zero_grad()
        #
        #     self.update_counter += 1
        # else:
        #     self.original_optimizer.zero_grad()
        #     self.optimizer_alpha.zero_grad()

        loss = f + torch.nn.functional.softplus(self.alpha) * g
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if self.batch_size_multiplier is not None and self.batch_size_multiplier > 1:
            if self.update_counter % self.batch_size_multiplier == 0:
                self.original_optimizer.step()
                self.alpha.grad *= -1
                self.optimizer_alpha.step()
        else:
            self.original_optimizer.step()
            self.alpha.grad *= -1
            self.optimizer_alpha.step()


        if self.alpha.item() < -2:
            self.alpha.data = torch.full_like(self.alpha.data, -2)
        elif self.alpha.item() > 30:
            self.alpha.data = torch.full_like(self.alpha.data, 30)
        return loss



