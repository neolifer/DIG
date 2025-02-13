
import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid
from torch.nn import Linear, ReLU, LayerNorm, BatchNorm1d
import math
import sys
import torch.nn.functional as F
sys.path.append('../')
sys.path.append('../..')
from torch.optim import Adam
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import tqdm
from torch_geometric.utils import to_networkx
from .pgexplainer import k_hop_subgraph_with_default_whole_graph
import networkx as nx
import time
from ..models.utils import LagrangianOptimization
from torch_geometric.nn import MessagePassing
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import add_self_loops, remove_self_loops, add_remaining_self_loops
import pickle as pk



writer = SummaryWriter()


# 
# class Temp_data(Data):
#     def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
#                  pos=None, normal=None, face=None, node_idx = None,**kwargs):
#         super(Data).__init__()
#         self.node_idx = node_idx
#     def __inc__(self, key, value):
#         if key == 'node_idx':
#             return self.node_idx.size(0)
#         else:
#             return super(Data).__inc__(key, value)
class Temp_data(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, node_index = None, subset = None, real_pred = None,**kwargs):
        super(Temp_data, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.node_index = node_index
        self.subset = subset
        self.real_pred = real_pred
class HardConcrete(torch.nn.Module):

    def __init__(self, beta=1 , gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3):
        super(HardConcrete, self).__init__()
        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)

        self.loc_bias = loc_bias

    def forward(self, input_element, summarize_penalty=True, training = True):
        input_element = input_element + self.loc_bias

        if self.training:
            u = torch.empty_like(input_element).uniform_(1e-6, 1.0-1e-6)

            s = sigmoid((input_element) / self.temp)
            # s = sigmoid((torch.log(u) - torch.log(1 - u) + input_element) / self.temp)
            penalty = sigmoid(input_element)
            # penalty = sigmoid(input_element - self.temp * self.gamma_zeta_ratio)

        else:
            s = sigmoid(input_element)
            penalty = torch.zeros_like(input_element)

        if summarize_penalty:
            penalty = penalty.mean()


        # s = s * (self.zeta - self.gamma) + self.gamma

        clipped_s = self.clip(s)
        if not training:
            return clipped_s, penalty
        #
        hard_concrete = (clipped_s >= 0.5).float()
        clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)


# def calculate_selected_nodes(data, edge_mask, top_k):
#     threshold = float(edge_mask.reshape(-1).sort(descending=True).values[min(top_k, edge_mask.shape[0]-1)])
#     hard_mask = (edge_mask > threshold).cpu()
#     edge_idx_list = torch.where(hard_mask == 1)[0]
#     selected_nodes = []
#     edge_index = data.edge_index.cpu().numpy()
#
#     for edge_idx in edge_idx_list:
#         selected_nodes += [edge_index[0][edge_idx], edge_index[1][edge_idx]]
#     selected_nodes = list(set(selected_nodes))
#     return selected_nodes



class GraphMaskAdjMatProbe(torch.nn.Module):

    device = None

    def __init__(self, vertex_embedding_dims, message_dims, n_class, h_dims):
        super(GraphMaskAdjMatProbe, self).__init__()
        self.n_class = n_class
        self.hard_gates = HardConcrete(fix_temp=True)

        transforms = []
        baselines = []
        for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
            transform_src = torch.nn.Sequential(
                Linear(v_dim, h_dim),
                BatchNorm1d(h_dim),
                ReLU(),
                Linear(h_dim, 1),
            )

            transforms.append(transform_src)

            baseline = torch.FloatTensor(m_dim)
            stdv = 1. / math.sqrt(m_dim)
            baseline.uniform_(-stdv, stdv)
            baseline = torch.nn.Parameter(baseline, requires_grad=False)

            baselines.append(baseline)

        transforms = torch.nn.ModuleList(transforms)
        self.transforms = transforms

        baselines = torch.nn.ParameterList(baselines)
        self.baselines = baselines
        self.attention = False
        # Initially we cannot update any parameters. They should be enabled layerwise
        for parameter in self.parameters():
            parameter.requires_grad = False

    def enable_layer(self, layer):
        # print("Enabling layer "+str(layer), file=sys.stderr)
        for parameter in self.transforms[layer].parameters():
            parameter.requires_grad = True

        self.baselines[layer].requires_grad = True

    def forward(self, gnn, training = True):
        latest_vertex_embeddings = gnn.get_latest_vertex_embedding()
        # gnn.latest_vertex_embeddings = None
        gates = []
        total_penalty = 0
        penalty_counts = 0
        for i in range(len(self.transforms)):
            if self.baselines[i].requires_grad == False:
                if training:
                    latest_vertex_embeddings[i] = None
                    gates.append(None)
                    continue
            srcs = latest_vertex_embeddings[i]
            latest_vertex_embeddings[i] = None
            if self.attention:
                srcs = srcs.reshape(srcs.shape[0], srcs.shape[1]*srcs.shape[2])
            # print(srcs.shape)
            # print([e.shape for e in self.transforms[i].parameters()])
            transformed_src = self.transforms[i](srcs)
            if training:

                gate, penalty = self.hard_gates(transformed_src.squeeze(-1), summarize_penalty=True, training = training)
                # penalty_norm = float(srcs.shape[0])
                # penalty = penalty.sum() / (penalty_norm + 1e-8)
                # penalty = penalty.sum()
                gates.append(gate)
                total_penalty += penalty
            else:
                gate, penalty = self.hard_gates(transformed_src.squeeze(-1), summarize_penalty=True, training = False)
                # penalty_norm = float(srcs.shape[0])
                # penalty = penalty.sum() / (penalty_norm + 1e-8)
                # penalty = penalty.sum()
                gates.append(gate)
                total_penalty += penalty

        return gates, self.baselines, total_penalty

    def set_device(self, device):
        self.to(device)


class GraphMaskExplainer(torch.nn.Module):
    def __init__(self, model, graphmask, epoch = 10, penalty_scaling = 0.5, entropy_scale = 1,
                 allowance = 0.03, lr1 =3e-4, lr2 = 3e-3, num_hops = None, batch_size =  1, use_baseline = False):
        super(GraphMaskExplainer, self).__init__()
        self.model = model
        self.graphmask = graphmask
        self.epoch = epoch
        self.device = 'cuda:0'
        self.loss = torch.nn.NLLLoss(reduction="none")
        # self.loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.penalty_scaling = penalty_scaling
        self.entropy_scale = entropy_scale
        self.allowance = allowance
        self.lr1 = lr1
        self.lr2 = lr2
        self.batch_size = batch_size
        self.num_hops = self.update_num_hops(num_hops)
        self.use_baseline = use_baseline
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

    def get_subgraph(self, node_idx, x, edge_index, y, **kwargs):

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



    def train_graphmask(self, dataset, dataset_name, explain_node_index_list):


        optimizer = Adam(self.graphmask.parameters(), lr=self.lr1, weight_decay=1e-5)
        decayRate = 0.96
        data = dataset[0]
        data = data.to(self.device)
        lagrangian_optimization = LagrangianOptimization(optimizer,
                                                         self.device,
                                                         alpha_optimizer_lr=self.lr2
                                                         )

        my_lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer= optimizer, step_size= 1000,gamma=0.1)
        my_lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer= lagrangian_optimization.optimizer_alpha,step_size=1000, gamma=0.1)
        with torch.no_grad():
            self.model.eval()

            # try:
            #     datalist = torch.load(f'checkpoints/graphmask_sub/graphmask_{dataset_name}_sub_train.pt')
            # except:
            datalist = []
            # large_index = pk.load(open('large_subgraph_bacom.pk','rb'))['node_idx']
            # motif = pk.load(open('Ba_Community_motif.plk','rb'))
            # explain_node_index_list = list(set(large_index).intersection(set(motif.keys())))
            # explain_node_index_list = torch.where(data.x)[0]
            # explain_node_index_list = list(range(len(data.train_mask)))
            # explain_node_index_list = torch.where(data.test_mask)[0]
            # explain_node_index_list = pk.load(open(f'{dataset_name}_within_nodes.pk','rb'))

            probs = []
            sizes = []
            for node_idx in tqdm.tqdm(explain_node_index_list):
                # for node_idx in explain_node_index_list:
                x, edge_index, y, subset, _ = \
                    self.get_subgraph(node_idx=node_idx, x=data.x, edge_index=data.edge_index, y=data.y)
                new_node_idx = torch.where(subset == node_idx)[0]
                datalist.append(Temp_data(x = x.cpu(), edge_index = edge_index.cpu(), node_index = torch.LongTensor([new_node_idx]).cpu()))
                # torch.save(datalist,f'checkpoints/graphmask_sub/graphmask_{dataset_name}_sub_train.pt')
            #     probs.append(F.softmax(self.model(x, edge_index)[new_node_idx], dim=-1).max(-1).values[0].cpu().data)
            #     sizes.append(edge_index.shape[-1])
            # return probs, sizes
        # explain_node_index_list = torch.where(data.test_mask)[0]
        # datalist = datalist = torch.load(f'checkpoints/graphmask_sub/graphmask_{dataset_name}_sub_train.pt')
        data = None
        loader = DataLoader(datalist, batch_size=self.batch_size, shuffle= True)

        for layer in reversed(list(range(len(self.model.convs)))):
            self.graphmask.enable_layer(layer)
            duration = 0.0
            if layer == 0:
                self.epoch += 1500
            for epoch in tqdm.tqdm(range(self.epoch)):
                # for epoch in range(self.epoch):
                if layer == 0:
                    my_lr_scheduler1.step()
                    my_lr_scheduler2.step()
                loss = 0.0
                self.graphmask.train()
                self.model.eval()
                tic = time.perf_counter()

                for i, batch in enumerate(loader):
                    # for batch in loader:
                    self.model.set_get_vertex(True)

                    x, edge_index, node_idx = batch.x, batch.edge_index, batch.node_index

                    x = x.to('cuda:0')
                    edge_index = edge_index.to('cuda:0')
                    node_idx = node_idx.to('cuda:0')
                    with torch.no_grad():
                        logits = self.model(x, edge_index)
                        real_pred = logits.argmax(-1).detach()
                        pred = F.log_softmax(logits, dim=-1)
                        real_loss = self.loss(pred[node_idx], real_pred[node_idx])
                        # real_pred = F.softmax(logits, dim = -1).detach()
                    gates, baselines, total_penalty = self.graphmask(self.model)
                    real_baseline = []

                    for i in range(len(gates)):
                        if baselines[i].requires_grad == False:
                            gates[i] = None
                            real_baseline.append(None)
                        else:
                            real_baseline.append(baselines[i])

                    self.model.set_get_vertex(False)
                    if self.use_baseline:
                        logits = self.model(x,edge_index, gates, real_baseline)
                    else:
                        logits = self.model(x,edge_index, gates)
                    pred = F.log_softmax(logits, dim=-1)
                    self.model.set_get_vertex(True)
                    # print(real_pred.shape, pred.shape)
                    # sys.exit()
                    loss_temp = self.loss(pred[node_idx], real_pred[node_idx]) - real_loss
                    g = torch.relu(loss_temp - self.allowance).mean()
                    # entropy = 0
                    # for gate in gates:
                    #     if gate is not None:
                    #         entropy +=  - gate * torch.log(gate) - (1 - gate) * torch.log(1 - gate)
                    f = total_penalty*self.penalty_scaling
                    loss2 = lagrangian_optimization.update(f, self.entropy_scale*g, self.graphmask)
                    # f.backward()
                    # optimizer.step()
                    # optimizer.zero_grad()

                    loss += g.detach().item()
                    # # loss_temp.backward()
                    # # optimizer.step()
                    # # optimizer.zero_grad()
                    # # loss += loss_temp.detach().item()
                    # # print(real_pred[node_idx])

                # for i in range(len(gates)):
                #     if self.graphmask.baselines[i].requires_grad == True:
                #         print(f'layer{i}:',torch.sum(gates[i].detach()/gates[i].shape[-1], dim=-1),total_penalty)

                duration += time.perf_counter() - tic


                # print(f'Layer: {layer} Epoch: {epoch} | Loss: {loss/(len(explain_node_index_list)/self.batch_size) }')

                # for i in reversed(list(range(3))):
                #     writer.add_scalar(f'Gate{epoch}{i}/train', gates[i].sum().detach().item(), epoch)
                # writer.add_scalar(f'Loss{layer}/train', loss / len(data.train_mask), epoch)
            # print(f"training time is {duration:.5}s")

    def forward(self,explanation_confidence,x, edge_index, new_node_idx,y, **kwargs):

        visualize = kwargs.get('visualize') if kwargs.get('visualize') is not None else False

        # y = kwargs.get('y')
        # x = data.x.to(self.device)
        #
        # edge_index = data.edge_index.to(self.device)
        # y = y.to(self.device)
        self.model.eval()
        self.graphmask.eval()


        # original value


        # masked value

        # x, edge_index, y, subset, _ = self.get_subgraph(node_idx, x, edge_index,y)
        # new_node_idx = torch.where(subset == node_idx)[0]
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        new_node_idx = new_node_idx.to(self.device)
        self.model.set_get_vertex(True)
        logits = self.model(x, edge_index)
        probs = F.softmax(logits, dim=-1)[new_node_idx].squeeze()
        # label = y[new_node_idx]
        label = probs.argmax(-1)
        gates, baselines, total_penalty = self.graphmask(self.model, training = False)
        origin = probs[label]

        # masked_pred = self.set_mask(x, edge_index, gates, new_node_idx,baselines)[label]
        # confidence = 1 - torch.abs(origin - masked_pred)/origin
        # related_preds = {}
        # related_preds['confidence'] = confidence
        related_preds = []
        top_k = 0
        sparsity = 1.0
        gate = [[],[]]

        # for i in range(len(gates)):
        #     gate[i] = torch.clone(gates[i])
        #     hard_concrete = (gate[i] >= 0.5).float()
        #     gate[i] = gate[i] + (hard_concrete - gate[i]).detach()
        # gates[0] = gates[-1]
        top_k = 0
        # for i in range(len(gates)):
        #     hard_concrete = (gates[i] >= 0.5).float()
        #     gates[i] = gates[i] + (hard_concrete - gates[i]).detach()
        # masked_pred = self.set_mask(x, edge_index, gates, new_node_idx, baselines)[label]
        # confidence = 1 - torch.abs(origin - masked_pred)/origin
        # print(confidence, (gates[-1].sum()/gates[-1].shape[0] + gates[0].sum()/gates[0].shape[0])/2)
        # sys.exit()
        confidence = 0
        for i in range(len(explanation_confidence)):

            if confidence >= explanation_confidence[i]:
                related_preds.append({
                    'explanation_confidence':explanation_confidence[i],
                    'sparsity': 1- top_k/gates[-1].shape[0]})
                continue
            while confidence < explanation_confidence[i]:
                # top_k = int((1- sparsity)*gates[-1].shape[0])

                ones = [torch.topk(gates[i], k= top_k, dim=0) for i in range(len(gates))]
                mask = [torch.zeros_like(gates[i]) for i in range(len(gates))]
                for k in range(len(gates)):
                    mask[k][ones[k].indices] = 1
                if self.use_baseline:
                    masked_pred = self.set_mask(x, edge_index, mask, new_node_idx,baselines)[label]
                else:
                    masked_pred = self.set_mask(x, edge_index, mask, new_node_idx)[label]
                confidence = 1 - torch.abs(origin - masked_pred)/origin

                if confidence >= explanation_confidence[i]:
                    related_preds.append({
                        'explanation_confidence':explanation_confidence[i],
                        'sparsity': 1- top_k/gates[-1].shape[0]})
                    break
                # elif top_k >= gates[-1].shape[0]:
                #     print('erro')
                #     sys.exit()
                else:
                    top_k += 1
                    # print(1)


                    # sparsity -= 0.05
        # top_k = [59, 70]
        # ones = [torch.topk(gates[i], k= top_k[i], dim=0) for i in range(len(gates))]
        # mask = [torch.zeros_like(gates[i]) for i in range(len(gates))]
        # for k in range(len(gates)):
        #     mask[k][ones[k].indices] = 1
        # for i in range(len(gates)):
        #     hard_concrete = (gates[i] >= 0.5).float()
        #     gates[i] = gates[i] + (hard_concrete - gates[i]).detach()
        # #     print(gates[i].sum())
        # related_preds = {}
        # # for i in range(len(gates)):
        # #     print((gates[i] == mask[i]).sum()/gates[i].shape[0])
        # # sys.exit()
        # masked_pred = self.set_mask(x, edge_index, gates, new_node_idx, baselines)[label]
        # confidence = 1 - torch.abs(origin - masked_pred)/origin
        # spars = 0
        # for i in range(len(gates)):
        #     spars += 1 - gates[i].sum()/gates[i].shape[0]
        # # # print(confidence, spars/2)
        # related_preds['evaluation_confidence'] = confidence
        # related_preds['sparsity'] = spars/2
        # print(related_preds)
        if not visualize:
            return related_preds
        return data, subset, new_node_idx, gates[-1]

    def set_mask(self, x, edge_index, gates, node_idx, baselines = None):

        self.model.set_get_vertex(False)
        logits = self.model(x,edge_index, gates, baselines)
        prob = F.softmax(logits, dim = -1)[node_idx].squeeze()
        self.model.set_get_vertex(True)
        return prob

    def train_explain_single(self, x, edge_index, new_node_idx, y,model_name,new_y):
        optimizer = Adam(self.graphmask.parameters(), lr=self.lr1, weight_decay=1e-5)
        decayRate = 0.96
        lagrangian_optimization = LagrangianOptimization(optimizer,
                                                         self.device,
                                                         alpha_optimizer_lr=self.lr2
                                                         )

        my_lr_scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer= optimizer, step_size= 1000,gamma=0.1)
        my_lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer= lagrangian_optimization.optimizer_alpha,step_size=1000, gamma=0.1)
        a = 0.50
        b = 0.05
        c = []
        while a < 1:
            c.append(a)
            a += b


        self.graphmask.train()
        self.model.eval()

        x = x.to('cuda:0')
        edge_index = edge_index.to('cuda:0')
        node_idx = new_node_idx.to('cuda:0')
        self.model.set_get_vertex(True)
        with torch.no_grad():
            logits = self.model(x, edge_index)
            probs = F.softmax(logits, dim=-1)[node_idx].squeeze()
            label = probs.argmax(-1)
            real_pred = logits.argmax(-1).detach()
            pred = F.log_softmax(logits, dim=-1)
            real_loss = self.loss(pred[node_idx], real_pred[node_idx])
        self.model.set_get_vertex(False)
        for layer in reversed(list(range(len(self.model.convs)))):
            self.graphmask.enable_layer(layer)
            duration = 0.0
            if layer == 0:
                if not 'GCN2' in model_name:
                    self.epoch += 100
            for epoch in range(self.epoch):
                # for epoch in range(self.epoch):
                if layer == 0:
                    my_lr_scheduler1.step()
                    my_lr_scheduler2.step()

                gates, baselines, total_penalty = self.graphmask(self.model)
                real_baseline = []

                for i in range(len(gates)):
                    if baselines[i].requires_grad == False:
                        gates[i] = None
                        real_baseline.append(None)
                    else:
                        real_baseline.append(baselines[i])
                if self.use_baseline:
                    logits = self.model(x,edge_index, gates, real_baseline)
                else:
                    logits = self.model(x,edge_index, gates)
                pred = F.log_softmax(logits, dim=-1)
                loss_temp = self.loss(pred[node_idx], real_pred[node_idx]) - real_loss
                g = torch.relu(loss_temp - self.allowance).mean()
                f = total_penalty*self.penalty_scaling
                loss2 = lagrangian_optimization.update(f, self.entropy_scale*g, self.graphmask)
        # for i in range(len(gates)):
        #     if self.graphmask.baselines[i].requires_grad == True:
        #         print(f'layer{i}:',torch.sum(gates[i].detach()/gates[i].shape[-1], dim=-1),total_penalty)
        # print(f'Layer: {layer} Epoch: {epoch} | Loss: {g}')
        self.graphmask.eval()
        gates, baselines, total_penalty = self.graphmask(self.model, training = False)
        origin = probs[label]
        related_preds = []
        top_k = 0
        # for i in range(len(gates)):
        #     hard_concrete = (gates[i] >= 0.5).float()
        #     gates[i] = gates[i] + (hard_concrete - gates[i]).detach()
        # masked_pred = self.set_mask(x, edge_index, gates, new_node_idx, baselines)[label]
        # confidence = 1 - torch.abs(origin - masked_pred)/origin
        # print(confidence, (gates[-1].sum()/gates[-1].shape[0] + gates[0].sum()/gates[0].shape[0])/2)
        # sys.exit()
        # confidence = 0
        # explanation_confidence = c
        # for i in range(len(explanation_confidence)):
        #
        #     if confidence >= explanation_confidence[i]:
        #         related_preds.append({
        #             'explanation_confidence':explanation_confidence[i],
        #             'sparsity': 1- top_k/gates[-1].shape[0]})
        #         continue
        #     while confidence < explanation_confidence[i]:
        #         # top_k = int((1- sparsity)*gates[-1].shape[0])
        #         ones = [torch.topk(gates[i], k= top_k, dim=0) for i in range(len(gates))]
        #         mask = [torch.zeros_like(gates[i]) for i in range(len(gates))]
        #         for k in range(len(gates)):
        #             mask[k][ones[k].indices] = 1
        #         if self.use_baseline:
        #             masked_pred = self.set_mask(x, edge_index, mask, new_node_idx, baselines)[label]
        #         else:
        #             masked_pred = self.set_mask(x, edge_index, mask, new_node_idx)[label]
        #         confidence = 1 - torch.abs(origin - masked_pred)/origin
        #
        #         if confidence >= explanation_confidence[i]:
        #             related_preds.append({
        #                 'explanation_confidence':explanation_confidence[i],
        #                 'sparsity': 1- top_k/gates[-1].shape[0]})
        #             break
        #
        #         else:
        #             top_k += 1
        # return related_preds
        # origin = probs
        # sparsities = []
        # s = 0
        # while s < 0.45:
        #     sparsities.append(0.5+s)
        #     s += 0.05
        # related_preds = {'fidelity':[],'acc':[]}
        # for s in sparsities:
        #     top_k = int((1- s)*gates[-1].shape[0])
        #     ones = [torch.topk(gates[i], k= top_k, dim=0) for i in range(len(gates))]
        #     mask = [torch.ones_like(gates[i]) for i in range(len(gates))]
        #     for k in range(len(gates)):
        #         mask[k][ones[k].indices] = 0
        #     if self.use_baseline:
        #         masked_pred = self.set_mask(x, edge_index, mask, new_node_idx, baselines)
        #     else:
        #         masked_pred = self.set_mask(x, edge_index, mask, new_node_idx)
        #     fidelity = origin[label] - masked_pred[label]
        #     ori_acc = origin.argmax() == y
        #     masked_acc = masked_pred.argmax() == y
        #     acc = masked_acc.item() - ori_acc.item()
        #     related_preds['fidelity'].append(fidelity.item())
        #     related_preds['acc'].append(acc)
        # return related_preds
        # edge_mask = F.softmax(edge_mask, dim=-1)
        edge_index = add_remaining_self_loops(edge_index)[0]
        for i in range(len(gates)):
            gates[i] = F.softmax(gates[i], dim=-1)
        class_count = [0,0,0,0,0,0,0]
        for j in range(len(gates)):
            edge_mask = gates[j]
            for i in range(edge_mask.shape[0]):
                class_count[new_y[edge_index[0,i]]] += edge_mask[i].item()/2
                class_count[new_y[edge_index[1,i]]] += edge_mask[i].item()/2
        for i in range(7):
            class_count[i] = class_count[i]/len(gates)
        # return self.hard_edge_mask, x, new_edge_index, edge_mask, related_preds
        return label.item(), class_count