import sys
sys.path.append('../../../../')
# from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.models import GCN2
import torch
import torch.nn as nn
from torch.optim import Adam
import shutil
import numpy as np
from torch.nn import functional as F
import os
import argparse
from load_dataset import  get_dataset, get_dataloader

def train_NC(args):
    GNNs = {'GCN2': GCN2}
    print('start loading data====================')
    import pdb; pdb.set_trace()
    dataset = get_dataset(args)
    input_dim = dataset.num_node_features
    output_dim = int(dataset.num_classes)

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', f"{args.dataset_name}")):
        os.mkdir(os.path.join('checkpoint', f"{args.dataset_name}"))
    ckpt_dir = f"./checkpoint/{args.dataset_name}/"
    model_level = args.model_level, dim_node = input_dim
    dim_hidden = args.dim_hidden, num_classes=output_dim
    alpha = args.alpha, theta=args.theta, num_layers=args.num_layers
    shared_weights=args.shared_weights, dropout=args.dropout
    data = dataset[0]
    gnnNets_NC = GNNs[args.model](model_level, dim_node, dim_hidden, num_classes, alpha, theta, num_layers,
                                  shared_weights, dropout)
    gnnNets_NC
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(gnnNets_NC.parameters(), lr=args.lr, weight_decay=args.wd)

    best_val_loss = float('inf')
    best_acc = 0
    val_loss_history = []
    early_stop_count = 0
    for epoch in range(1, args.epoch + 1):
        gnnNets_NC.train()
        logits= gnnNets_NC(data)
        prob = F.softmax(logits, dim=-1)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        eval_info = evaluate_NC(data, gnnNets_NC, criterion)
        eval_info['epoch'] = epoch

        if eval_info['val_loss'] < best_val_loss:
            best_val_loss = eval_info['val_loss']
            val_acc = eval_info['val_acc']

        val_loss_history.append(eval_info['val_loss'])

        # only save the best model
        is_best = (eval_info['val_acc'] > best_acc)

        if eval_info['val_acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > args.early_stopping:
            break

        if is_best:
            best_acc = eval_info['val_acc']
        if is_best or epoch % args.save_epoch == 0:
            save_best(ckpt_dir, epoch, gnnNets_NC, args.model_name, eval_info['val_acc'], is_best)
            print(f'Epoch {epoch}, Train Loss: {eval_info["train_loss"]:.4f}, '
                        f'Train Accuracy: {eval_info["train_acc"]:.3f}, '
                        f'Val Loss: {eval_info["val_loss"]:.3f}, '
                        f'Val Accuracy: {eval_info["val_acc"]:.3f}')


    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{args.model_name}_best.pth'))
    gnnNets_NC.update_state_dict(checkpoint['net'])
    eval_info = evaluate_NC(data, gnnNets_NC, criterion)
    print(f'Test Loss: {eval_info["test_loss"]:.4f}, Test Accuracy: {eval_info["test_acc"]:.3f}')


def evaluate_NC(data, gnnNets_NC, criterion):
    eval_state = {}
    gnnNets_NC.eval()

    with torch.no_grad():
        for key in ['train', 'val', 'test']:
            mask = data['{}_mask'.format(key)]
            logits, probs, _ = gnnNets_NC(data)
            loss = criterion(logits[mask], data.y[mask]).item()
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            ## record
            eval_state['{}_loss'.format(key)] = loss
            eval_state['{}_acc'.format(key)] = acc

    return eval_state


def save_best(ckpt_dir, epoch, gnnNets, model_name, eval_acc, is_best):
    print('saving....')
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
    gnnNets.to_device()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model', default='GCN2', dest='gnn models')
    args.add_argument('--model_level', default='node')
    args.add_argument('--dim_hidden', default=128)
    args.add_argument('--alpha', default=0.1)
    args.add_argument('--theta', default=0.5)
    args.add_argument('--num_layers', default=6)
    args.add_argument('--shared_weights', default=False)
    args.add_argument('--dropout', default=0.5)
    args.add_argument('--dataset_dir', default='../datasets/')
    args.add_argument('--dataset_name', default='BA_shapes')
    args.add_argument('--epoch', default=500)
    args.add_argument('--lr', default=0.01)
    args.add_argument('--wd1', default=1e-3)
    args.add_argument('--wd2', default=1e-5)
    args = args.parse_args()
    train_NC(args)