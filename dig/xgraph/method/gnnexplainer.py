import sys

import torch
from torch import Tensor
from torch_geometric.utils.loop import add_self_loops, remove_self_loops, add_remaining_self_loops
from dig.version import debug
from ..models.utils import subgraph
from torch.nn.functional import cross_entropy
from .base_explainer import ExplainerBase
from torch.nn import functional as F
from torch_geometric.data import Data, Batch,DataLoader

EPS = 1e-15

def cross_entropy_with_logit(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return cross_entropy(y_pred, y_true.long(), **kwargs)

class GNNExplainer(ExplainerBase):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note:: For an example, see `benchmarks/xgraph
        <https://github.com/divelab/DIG/tree/dig/benchmarks/xgraph>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        explain_graph (bool, optional): Whether to explain graph classification model
            (default: :obj:`False`)
    """

    coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=1000, lr=0.01, explain_graph=False):
        super(GNNExplainer, self).__init__(model, epochs, lr, explain_graph)
        self.loss = torch.nn.NLLLoss()


    def __loss__(self, raw_preds, x_label):
        if self.explain_graph:
            loss = cross_entropy_with_logit(raw_preds, x_label)
        else:
            raw_preds = F.log_softmax(raw_preds, dim = -1)
            # print(raw_preds[self.node_idx].shape, x_label.unsqueeze(0).shape)
            # sys.exit()
            loss = self.loss(raw_preds[self.node_idx].unsqueeze(0), x_label.unsqueeze(0))

        m = self.edge_mask.sigmoid()
        temp = loss.item()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        if self.mask_features:
            m = self.node_feat_mask.sigmoid()
            loss = loss + self.coeffs['node_feat_size'] * m.sum()
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss, temp, m.sum().item()

    def gnn_explainer_alg(self,
                          x: Tensor,
                          edge_index: Tensor,
                          ex_label: Tensor,
                          mask_features: bool = False,
                          **kwargs
                          ) -> None:

        # initialize a mask
        self.to(x.device)
        self.mask_features = mask_features

        # train to get the mask
        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)
        for conv in self.model.convs:
            conv.require_sigmoid = True
        for epoch in range(1, self.epochs + 1):

            if mask_features:
                h = x * self.node_feat_mask.view(1, -1).sigmoid()
            else:
                h = x
            raw_preds = self.model(h, edge_index, **kwargs)
            loss,pred_loss, size_loss = self.__loss__(raw_preds, ex_label)
            if epoch % 20 == 0 and debug:
                print(f'Loss:{loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f'pred loss: {pred_loss}, size loss: {size_loss}')
        for conv in self.model.convs:
            conv.require_sigmoid = False
        return self.edge_mask.data

    def forward(self, x, edge_index, y,evaluation_confidence, mask_features=False, control_sparsity = False,**kwargs):
        r"""
        Run the explainer for a specific graph instance.

        Args:
            x (torch.Tensor): The graph instance's input node features.
            edge_index (torch.Tensor): The graph instance's edge index.
            mask_features (bool, optional): Whether to use feature mask. Not recommended.
                (Default: :obj:`False`)
            **kwargs (dict):
                :obj:`node_idx` （int): The index of node that is pending to be explained.
                (for node classification)
                :obj:`sparsity` (float): The Sparsity we need to control to transform a
                soft mask to a hard mask. (Default: :obj:`0.7`)
                :obj:`num_classes` (int): The number of task's classes.

        :rtype: (None, list, list)

        .. note::
            (None, edge_masks, related_predictions):
            edge_masks is a list of edge-level explanation for each class;
            related_predictions is a list of dictionary for each class
            where each dictionary includes 4 type predicted probabilities.

        """
        super().forward(x=x, edge_index=edge_index, **kwargs)
        self.model.eval()

        # self_loop_edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=self.num_nodes)

        # Only operate on a k-hop subgraph around `node_idx`.
        # Get subgraph and relabel the node, mapping is the relabeled given node_idx.
        # if not self.explain_graph:
        node_idx = kwargs.get('node_idx')
        self.node_idx = node_idx
        #     assert node_idx is not None
        subset, new_edge_index, inv, self.hard_edge_mask = subgraph(
            node_idx, self.__num_hops__, edge_index, relabel_nodes=True,
            num_nodes=None, flow=self.__flow__())
        # return None,None,None,None, new_edge_index.shape[-1]
        if new_edge_index.shape[-1] <= 1:
            return False
        new_x = x[subset]
        # new_y = y[subset]
        new_node_index = int(torch.where(subset == node_idx)[0])
        self.node_idx = new_node_index
        # Assume the mask we will predict
        new_edge_index = add_remaining_self_loops(new_edge_index)[0]

        self.model.to(x.device)
        label = F.softmax(self.model(new_x, new_edge_index), dim = -1)[new_node_index].squeeze().argmax(-1)
        # Calculate mask


        self.__clear_masks__()
        self.__set_masks__(new_x, new_edge_index)


        edge_mask= self.gnn_explainer_alg(new_x, new_edge_index, label)
        related_preds = []
        sparsity = 1
        confidence = 0
        for e in evaluation_confidence:
            if confidence >= e:
                related_preds.append(sparsity)
                continue
            while confidence < e:
                temp_mask = self.control_sparsity(edge_mask, sparsity = sparsity)
                with torch.no_grad():
                    temp = self.eval_related_pred(new_x, new_edge_index, temp_mask, label, **kwargs)
                confidence = temp[0]['evaluation_confidence'].item()
                if confidence >= e:
                    related_preds.append(sparsity)
                    break
                else:
                    sparsity -= 0.05
        self.__clear_masks__()

        return self.hard_edge_mask, x, new_edge_index, edge_mask, related_preds




    def __repr__(self):
        return f'{self.__class__.__name__}()'