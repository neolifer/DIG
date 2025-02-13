import sys
sys.path.append('../../../../')
# from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.models import *
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import shutil
import numpy as np
from torch.nn import functional as F
import os
import argparse
from load_dataset import  get_dataset, get_dataloader
import sys
from tqdm import tqdm
from torch_geometric.datasets import CitationFull, Planetoid
import torch_geometric.transforms as T


def train_NC(parser,  lr ,head, dropout, wd2, hid_dim):
    GNNs = {'GCN2': GCN2}
    # print('start loading data====================')
    # import pdb; pdb.set_trace()
    if parser.dataset_name != 'Cora':
        dataset = get_dataset(parser)
    else:
        dataset = Planetoid('../datasets', 'Cora',split="public")
    dataset.data.x = dataset.data.x.to(torch.float32)
    # dataset.data.x = dataset.data.x[:, :1]
    input_dim = dataset.num_node_features


    output_dim = int(dataset.num_classes)

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', f"{parser.dataset_name}")):
        os.mkdir(os.path.join('checkpoint', f"{parser.dataset_name}"))
    ckpt_dir = f"./checkpoint/{parser.dataset_name}/"
    model_level = parser.model_level
    dim_node = input_dim
    dim_hidden = parser.dim_hidden
    num_classes=output_dim
    alpha = parser.alpha
    theta=parser.theta
    num_layers=parser.num_layers
    shared_weights=parser.shared_weights
    wd1 = parser.wd1


    data = dataset.data

    # gnnNets_NC = GM_GCN2(model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
    #                               shared_weights, dropout)
    # for conv in gnnNets_NC.convs:
    #     conv.get_vertex = True
    # gnnNets_NC = GCN_2l(model_level, dim_node, dim_hidden, num_classes)
    gnnNets_NC = GM_GCN(num_layers, dim_node, hid_dim, num_classes, dropout)
    # gnnNets_NC = GAT(num_layers, dim_node, hid_dim, num_classes, dropout, heads = head)
    # gnnNets_NC = GraphSAGE(num_layers, dim_node, dim_hidden, num_classes)
    gnnNets_NC = gnnNets_NC.cuda()
    criterion = nn.NLLLoss()
    # reg_params = list(gnnNets_NC.convs.parameters())
    # non_reg_params = list(gnnNets_NC.fcs.parameters())
    # optimizer = torch.optim.Adam([
    #     dict(params=reg_params, weight_decay= wd1),
    #     dict(params=non_reg_params, weight_decay=wd2)
    # ], lr=lr)
    optimizer = torch.optim.Adam(params=gnnNets_NC.parameters(), weight_decay= wd2, lr=lr)

    best_val_loss = float('inf')
    best_acc = 0
    val_loss_history = []
    early_stop_count = 0
    data = data.cuda()
    stop_val_loss = float('inf')
    for epoch in tqdm(range(1, parser.epoch + 1)):
        gnnNets_NC.train()

        logits= gnnNets_NC(data.x, data.edge_index)
        prob = F.log_softmax(logits, dim=-1)

        loss = criterion(prob[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(gnnNets_NC.parameters(), clip_value=2)
        optimizer.step()
        # for name, param in list(gnnNets_NC.named_parameters()):
        #     if param.requires_grad:
        #         print(name, param)
        eval_info = evaluate_NC(data, gnnNets_NC, criterion)
        eval_info['epoch'] = epoch

        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            val_acc = eval_info['val_acc']

        val_loss_history.append(eval_info['val_loss'])

        # only save the best model
        is_best = (eval_info['val_acc'] >= best_acc)

        if eval_info['val_loss'] < stop_val_loss:
            early_stop_count = 0
            stop_val_loss = eval_info['val_loss']
        else:
            early_stop_count += 1

        if early_stop_count > parser.early_stopping:
            break

        if is_best:
            best_acc = eval_info['val_acc']
        if is_best or epoch % parser.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets_NC, parser.model_name, eval_info['val_acc'], is_best)
            print(f'Epoch {epoch}, Train Loss: {eval_info["train_loss"]:.4f}, '
                        f'Train Accuracy: {eval_info["train_acc"]:.3f}, '
                        f'Val Loss: {eval_info["val_loss"]:.3f}, '
                        f'Val Accuracy: {eval_info["val_acc"]:.3f}',
                        f'Test Accuracy: {eval_info["test_acc"]:.3f}')


    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{parser.model_name}_best.pth'))
    gnnNets_NC.load_state_dict(checkpoint['net'])
    eval_info = evaluate_NC(data, gnnNets_NC, criterion)
    print(f'Test Loss: {eval_info["test_loss"]:.4f}, Test Accuracy: {eval_info["test_acc"]:.3f}')
    return eval_info["test_acc"]

def evaluate_NC(data, gnnNets_NC, criterion):
    eval_state = {}
    gnnNets_NC.eval()

    with torch.no_grad():
        for key in ['train', 'val', 'test']:
            mask = data['{}_mask'.format(key)]
            logits= gnnNets_NC(data.x, data.edge_index)
            probs = F.log_softmax(logits, dim =-1)
            loss = criterion(probs[mask], data.y[mask]).item()
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            ## record
            eval_state['{}_loss'.format(key)] = loss
            eval_state['{}_acc'.format(key)] = acc

    return eval_state


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    # print('saving....')
    gnnNets.to('cpu')
    state = {
        'net': gnnNets.state_dict(),
        'epoch': epoch,
        'acc': eval_acc
    }
    pth_name = f"{model_name}_latest.pth"
    best_pth_name = f'{model_name}_best.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    gnnNets.cuda()

class ARGS():
    def __init__(self, ps):
        self.model = ps.model
        self.model_level = ps.model_level
        self.dim_hidden = ps.dim_hidden
        self.alpha = ps.alpha
        self.theta = ps.theta
        self.num_layers = ps.num_layers
        self.shared_weights = ps.shared_weights
        self.dropout = ps.dropout
        self.dataset_dir = ps.dataset_dir
        self.dataset_name = ps.dataset_name
        self.epoch = ps.epoch
        self.lr = ps.lr
        self.wd1 = ps.wd1
        self.wd2 = ps.wd2
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='GCN2', dest='gnn models')
    parser.add_argument('--model_name', default='GCN_nopre')
    parser.add_argument('--model_level', default='node')
    parser.add_argument('--dim_hidden', default=64)
    parser.add_argument('--alpha', default=0.1)
    parser.add_argument('--theta', default=0.5)
    parser.add_argument('--num_layers', default=2)
    parser.add_argument('--shared_weights', default=False)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--dataset_dir', default='../datasets/')
    parser.add_argument('--dataset_name', default='Cora')
    parser.add_argument('--epoch', default=1500)
    parser.add_argument('--save_epoch', default=10)
    parser.add_argument('--lr', default=0.01)
    parser.add_argument('--wd1', default=1e-2)
    parser.add_argument('--wd2', default=5e-3)
    parser.add_argument('--early_stopping', default=100)
    ps = parser.parse_args()
    heads = []
    import random
    import numpy as np
    torch.manual_seed(42)
    random.seed(0)
    np.random.seed(0)
    for a in range(1,9):
        for b in range(1,9):
                for c in range(1,4):
                    heads.append([a,b,c])
    heads = [[8,]]
    lrs = [1e-2]
    dropouts = [0.7]
    wd2s = [1e-3]
    hid_dims = [64]
    best_acc = 0
    best_parameters = []
    from itertools import product
    for lr, head, dropout, wd2, hid_dim in tqdm(iterable= product(lrs, heads, dropouts, wd2s, hid_dims), total= len(list(product(lrs, heads, dropouts, wd2s, hid_dims)))):
        print(f'lr: {lr}', f'head: {head}', f'dropout: {dropout}', f'wd2: {wd2}', f'hid_dim: {hid_dim}')
        acc = train_NC(ps, lr ,head, dropout, wd2, hid_dim)
        if acc > best_acc:
            best_parameters = [lr, head, dropout, wd2, hid_dim]
            best_acc = acc
            print('new best:',best_acc, f'lr: {lr}', f'head: {head}', f'dropout: {dropout}', f'wd2: {wd2}', f'hid_dim: {hid_dim}')
    print(best_parameters, best_acc)