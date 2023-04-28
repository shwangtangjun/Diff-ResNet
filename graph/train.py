import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from utils import load_data, accuracy
from model import DiffusionNet
import copy

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--num_splits', type=int, default=100, help='Number of different splits.')
parser.add_argument('--num_inits', type=int, default=20, help='Number of different initializations.')
parser.add_argument('--device', type=str, default='0')

parser.add_argument('--max_epochs', type=int, default=10000, help='Max uumber of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--patience', type=int, default=50, help='Early Stop Patience.')

parser.add_argument('--dataset', type=str, default="cora")
parser.add_argument('--step_size', type=float)
parser.add_argument('--layer_num', type=int)
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # specify which GPU(s) to be used


def train(model, optimizer, adj, features, labels, idx_train):
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)
    loss = nn.CrossEntropyLoss()(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()


def val(model, adj, features, labels, idx_val):
    model.eval()
    output = model(features, adj)
    loss = nn.CrossEntropyLoss()(output[idx_val], labels[idx_val])
    acc = accuracy(output[idx_val], labels[idx_val])
    loss = loss.detach().cpu().numpy()
    acc = acc.cpu().numpy()

    return loss, acc


def test(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj)
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return acc_test


def run_single_trial_of_single_split(adj, features, labels, idx_train, idx_val, idx_test, diagonal, torch_seeds):
    torch.manual_seed(torch_seeds)
    torch.cuda.manual_seed(torch_seeds)

    model = DiffusionNet(n_features=features.shape[1], num_classes=labels.max().item() + 1, step=args.step_size,
                         layer_num=args.layer_num, dropout=args.dropout, diagonal=diagonal.cuda())

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model = model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

    val_loss_min = np.inf
    val_acc_max = 0
    patience_step = 0
    best_state_dict = None

    val_loss_list = []
    val_acc_list = []
    for epoch in range(args.max_epochs):
        train(model, optimizer, adj, features, labels, idx_train)
        val_loss, val_acc = val(model, adj, features, labels, idx_val)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        if val_loss <= val_loss_min or val_acc >= val_acc_max:
            val_loss_min = np.min((val_loss, val_loss_min))
            val_acc_max = np.max((val_acc, val_acc_max))
            patience_step = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_step += 1

        if patience_step >= args.patience:
            model.load_state_dict(best_state_dict)
            break

    acc = test(model, adj, features, labels, idx_test)
    acc = acc.cpu().numpy()
    return acc


def run_single_split(seed):
    random_state = np.random.RandomState(seed)
    adj, features, labels, idx_train, idx_val, idx_test, diagonal = load_data(args.dataset, random_state)
    torch_seeds = random_state.randint(0, 1000000, args.num_inits)  # 20 trials for each split
    acc_list = []
    for i in range(args.num_inits):
        acc = run_single_trial_of_single_split(adj, features, labels, idx_train, idx_val, idx_test, diagonal,
                                               torch_seeds[i])
        acc_list.append(acc)
    return np.array(acc_list)


def main():
    random_state = np.random.RandomState(args.seed)
    single_split_seed = random_state.randint(0, 1000000, args.num_splits)  # 100 random splits

    total_acc_list = []
    for i in range(args.num_splits):
        acc_of_single_split = run_single_split(single_split_seed[i])
        print(acc_of_single_split)
        total_acc_list.append(acc_of_single_split)

    print(np.mean(total_acc_list) * 100)
    print(np.std(total_acc_list) * 100)
    print(args.dropout)
    print(args.step_size)
    print(args.layer_num)


if __name__ == '__main__':
    main()
