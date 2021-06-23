
import torch
from torch.nn.parameter import Parameter
import numpy as np
from torch import sigmoid
from torch.nn import Linear, ReLU
import math

class HardConcrete(torch.nn.Module):

    def __init__(self, beta=1 / 3, gamma=-0.2, zeta=1.0, fix_temp=True, loc_bias=3):
        super(HardConcrete, self).__init__()

        self.temp = beta if fix_temp else Parameter(torch.zeros(1).fill_(beta))
        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = np.math.log(-gamma / zeta)

        self.loc_bias = loc_bias

    def forward(self, input_element, summarize_penalty=True):
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

        if True:
            hard_concrete = (clipped_s > 0.5).float()
            clipped_s = clipped_s + (hard_concrete - clipped_s).detach()

        return clipped_s, penalty

    def clip(self, x, min_val=0, max_val=1):
        return x.clamp(min_val, max_val)


class GraphMaskAdjMatProbe(torch.nn.Module):

    device = None

    def __init__(self, vertex_embedding_dims, message_dims, n_relations, h_dims):
        super(GraphMaskAdjMatProbe, self).__init__()
        self.n_relations = n_relations
        self.hard_gates = HardConcrete()

        transforms = []
        baselines = []
        for v_dim, m_dim, h_dim in zip(vertex_embedding_dims, message_dims, h_dims):
            transform_src = torch.nn.Sequential(
                Linear(v_dim, h_dim),
                ReLU(),
                Linear(h_dim, m_dim * n_relations),
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

            src_shape = srcs.size()

            new_shape = list(src_shape)[:-1] + [self.n_relations, src_shape[-1]]
            transformed_src = self.transforms[i](srcs).view(new_shape).unsqueeze(1)

            tgt = srcs.unsqueeze(2).unsqueeze(2)

            a = (transformed_src * tgt).transpose(-2, 1)

            squeezed_a = a.sum(dim=-1)
            gate, penalty = self.hard_gates(squeezed_a, summarize_penalty=False)

            gate = gate * (adj_mat > 0).float()
            penalty_norm = (adj_mat > 0).sum().float()
            penalty = (penalty * (adj_mat > 0).float()).sum() / (penalty_norm + 1e-8)

            gates.append(gate)
            total_penalty += penalty

        return gates, self.baselines, total_penalty

    def set_device(self, device):
        self.to(device)