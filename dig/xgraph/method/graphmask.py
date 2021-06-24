
import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid
from torch.nn import Linear, ReLU
import math
import sys
import torch.functional as F
sys.path.append('../')
sys.path.append('../..')
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from pgexplainer import k_hop_subgraph_with_default_whole_graph, calculate_selected_nodes
import networkx as nx
import time
from models.utils import LagrangianOptimization
from .shapley import GnnNets_GC2value_func, GnnNets_NC2value_func, gnn_score

class temp_data():
    def __init__(self, x, edge_index):
        self.x = x
        self.edge_index = edge_index


class HardConcrete(torch.nn.Module):

    def __init__(self, beta=1 / 3, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3):
        super(HardConcrete, self).__init__()

        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)

        self.loc_bias = loc_bias

    def forward(self, input_element, summarize_penalty=True, train = False):
        input_element = input_element + self.loc_bias

        if self.training:
            u = torch.empty_like(input_element).uniform_(1e-6, 1.0-1e-6)

            s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) / self.temp)

            penalty = sigmoid(input_element - self.temp * self.gamma_zeta_ratio)
            penalty = penalty
        else:
            s = sigmoid(input_element)
            penalty = torch.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()

        s = s * (self.zeta - self.gamma) + self.gamma

        clipped_s = self.clip(s)
        if train:
            if True:
                hard_concrete = (clipped_s > 0.5).float()
                clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

            return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)


class GraphMaskAdjMatProbe(torch.nn.Module):

    device = None

    def __init__(self, vertex_embedding_dims, message_dims, n_class, h_dims):
        super(GraphMaskAdjMatProbe, self).__init__()
        self.n_class = n_class
        self.hard_gates = HardConcrete()

        transforms = []
        baselines = []
        for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
            transform_src = torch.nn.Sequential(
                Linear(v_dim, h_dim),
                ReLU(),
                Linear(h_dim, m_dim),
            )

            transforms.append(transform_src)

            baseline = torch.FloatTensor(m_dim)
            stdv = 1. / math.sqrt(m_dim)
            baseline.uniform_(-stdv, stdv)
            baseline = torch.nn.Parameter(baseline, requires_grad=True)

            baselines.append(baseline)

        transforms = torch.nn.ModuleList(transforms)
        self.transforms = transforms

        baselines = torch.nn.ParameterList(baselines)
        self.baselines = baselines

        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_layer(self, layer):
        print("Enabling layer "+str(layer), file=sys.stderr)
        for parameter in self.transforms[layer].parameters():
            parameter.requires_grad = True

        self.baselines[layer].requires_grad = True

    def forward(self, gnn):
        latest_vertex_embeddings = gnn.get_latest_vertex_embeddings()
        adj_mat = gnn.get_latest_adj_mat()

        gates = []
        total_penalty = 0
        for i in range(len(self.transforms)):
            srcs = latest_vertex_embeddings[i]


            transformed_src = self.transforms[i](srcs)


            gate, penalty = self.hard_gates(transformed_src, summarize_penalty=False)


            penalty_norm = srcs.shape[0].float()
            penalty = penalty.sum() / (penalty_norm + 1e-8)

            gates.append(gate)
            total_penalty += penalty

        return gates, self.baselines, total_penalty

    def set_device(self, device):
        self.to(device)


class GraphMaskExplainer(torch.nn.Module):
    def __init__(self, model, graphmask, epoch, penalty_scaling = 1, allowance = 0.03):
        super(GraphMaskExplainer, self).__init__()
        self.model = model
        self.graphmask = graphmask
        self.epoch = epoch
        self.device = 'cuda:0'
        self.loss = torch.nn.CrossEntropyLoss()
        self.penalty_scaling = penalty_scaling
        self.allowance = allowance

    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        graph = to_networkx(data=Data(x=x, edge_index=edge_index), to_undirected=True)

        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True,
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
        y = y[subset]
        return x, edge_index, y, subset, kwargs



    def train_graphmask(self, dataset):

        self.graphmask.train()
        optimizer = Adam(self.graphmask.parameters(), lr=self.lr, weight_decay=1e-5)
        data = dataset[0]
        dataset_indices = torch.where(data.train_mask != 0)[0].tolist()
        x_dict = {}
        edge_index_dict = {}
        node_idx_dict = {}
        pred_dict = {}
        with torch.no_grad():
            data = dataset[0]
            data.to(self.device)
            self.model.eval()
            x_dict = {}
            edge_index_dict = {}
            node_idx_dict = {}
            pred_dict = {}
            for node_idx in tqdm(torch.where(data.train_mask)[0].tolist()):
                x, edge_index, y, subset, _ = \
                    self.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
                logits = self.model(data)
                x_dict[node_idx] = x.to(self.device)
                edge_index_dict[node_idx] = edge_index.to(self.device)
                node_idx_dict[node_idx] = int(torch.where(subset == node_idx)[0])
                pred_dict[node_idx] = logits[node_idx_dict[node_idx]].argmax(-1).cpu()
        duration = 0.0
        lagrangian_optimization = LagrangianOptimization(optimizer,
                                                         self.device
                                                        )
        for epoch in range(10):
            loss = 0.0
            optimizer.zero_grad()
            tmp = float(self.t0 * np.power(self.t1 / self.t0, epoch / self.epochs))
            self.graphmask.train()
            tic = time.perf_counter()
            for iter_idx, node_idx in tqdm(enumerate(x_dict.keys())):
                data = temp_data(x_dict[node_idx], edge_index_dict[node_idx])
                gates, baselines, total_penalty = self.graphmask(self.model)
                logits = self.model(data, gates, baselines)
                pred = F.softmax(logits, dim=-1)
                loss_temp = self.loss(pred[node_idx_dict[node_idx]], pred_dict[node_idx])
                g = torch.relu(torch.relu(loss_temp - self.allowance).mean())
                f = total_penalty*self.penalty_scaling
                loss_temp = lagrangian_optimization.update(f, g)
                loss += loss_temp.item()

            optimizer.step()
            duration += time.perf_counter() - tic
            print(f'Epoch: {epoch} | Loss: {loss / len(x_dict)}')
        print(f"training time is {duration:.5}s")

    def forward(self,data, **kwargs):
        top_k = kwargs.get('top_k') if kwargs.get('top_k') is not None else 10

        y = kwargs.get('y')
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        y = y.to(self.device)
        self.model.eval()
        self.graphmask.eval()
        logits = self.model(data)
        probs = F.softmax(logits, dim=-1)
        node_idx = kwargs.get('node_idx')
        assert kwargs.get('node_idx') is not None, "please input the node_idx"
        # original value
        probs = probs.squeeze()[node_idx]
        label = y[node_idx]
        # masked value
        x, edge_index, _, subset, _ = self.get_subgraph(node_idx, x, edge_index)
        new_node_idx = torch.where(subset == node_idx)[0]
        data = temp_data(x, edge_index)
        gates, baselines, total_penalty = self.graphmask(self.model)
        data = Data(x=x, edge_index=edge_index)
        selected_nodes = calculate_selected_nodes(data, gates, top_k)
        maskout_nodes_list = [node for node in range(data.x.shape[0]) if node not in selected_nodes]
        value_func = GnnNets_NC2value_func(self.model,
                                           node_idx=new_node_idx,
                                           target_class=label)
        maskout_pred = gnn_score(maskout_nodes_list, data, value_func,
                                 subgraph_building_method='zero_filling')
        sparsity_score = 1 - len(selected_nodes) / data.x.shape[0]
        pred_mask = [gates]
        related_preds = [{
            'maskout': maskout_pred,
            'origin': probs[label],
            'sparsity': sparsity_score}]
        return None, pred_mask, related_preds