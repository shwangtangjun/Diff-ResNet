"""
Code to reproduce Table 3.
"""
import os
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
from utils import get_tqdm, get_configuration, get_dataloader, get_embedded_feature, get_base_mean
from utils import compute_confidence_interval, calculate_weight
from diffresnet import DiffusionResNet

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int, help='seed for training')
parser.add_argument("--dataset", choices=["mini", "tiered", "cub"], type=str)
parser.add_argument("--backbone", choices=["resnet18", "wideres"], type=str)
parser.add_argument("--query_per_class", default=15, type=int, help="number of unlabeled query sample per class")
parser.add_argument("--way", default=5, type=int, help="5-way-k-shot")
parser.add_argument("--test_iter", default=10000, type=int, help="test on 10000 tasks and output average accuracy")
parser.add_argument("--shot", choices=[1, 5], type=int)
parser.add_argument('--silent', action='store_true', help='call --silent to disable tqdm')

parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument("--step_size", type=float, help='strength of each diffusion layer', default=0.5)
parser.add_argument("--layer_num", type=int, help='number of diffusion layers, 0 means no diffusion')
parser.add_argument("--n_top", type=int, default=8)
parser.add_argument("--sigma", type=int, default=4)

parser.add_argument("--lamda", help='parameter in LaplacianShot', default=0.5, type=float)
parser.add_argument("--method", choices=['simple', 'laplacian', 'diffusion'], type=str)
parser.add_argument("--alpha", help='parameter for weighted sum of ce loss and proto loss', type=float, default=0.0)

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # specify which GPU(s) to be used


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    data_path, split_path, save_path, num_classes = get_configuration(args.dataset, args.backbone)

    # On novel class: get the output of embedding function (backbone)
    # On base class: get the output average of embedding function (backbone), used for centering
    train_loader = get_dataloader(data_path, split_path, 'train')
    test_loader = get_dataloader(data_path, split_path, 'test')
    embedded_feature = get_embedded_feature(test_loader, save_path, args.silent)
    base_mean = get_base_mean(train_loader, save_path, args.silent)

    acc_list = []
    tqdm_test_iter = get_tqdm(range(args.test_iter), args.silent)

    for _ in tqdm_test_iter:
        if args.method == 'simple':
            acc = simple_shot(embedded_feature, base_mean)
        elif args.method == 'laplacian':
            acc = laplacian_shot(embedded_feature, base_mean)
        elif args.method == 'diffusion':
            acc = single_trial(embedded_feature, base_mean)
        else:
            raise NotImplementedError

        acc_list.append(acc)

        if not args.silent:
            tqdm_test_iter.set_description('Test on few-shot tasks. Accuracy:{:.2f}'.format(np.mean(acc_list)))

    acc_mean, acc_conf = compute_confidence_interval(acc_list)
    print('Accuracy:{:.2f}'.format(acc_mean))
    print('Conf:{:.2f}'.format(acc_conf))


def sample_task(embedded_feature):
    """
    Sample a single few-shot task from novel classes
    """
    sample_class = random.sample(list(embedded_feature.keys()), args.way)
    train_data, test_data, test_label, train_label = [], [], [], []

    for i, each_class in enumerate(sample_class):
        samples = random.sample(embedded_feature[each_class], args.shot + args.query_per_class)

        train_label += [i] * args.shot
        test_label += [i] * args.query_per_class
        train_data += samples[:args.shot]
        test_data += samples[args.shot:]

    return np.array(train_data), np.array(test_data), np.array(train_label), np.array(test_label)


def single_trial(embedded_feature, base_mean):
    train_data, test_data, train_label, test_label = sample_task(embedded_feature)

    train_data, test_data, train_label, test_label, base_mean = torch.tensor(train_data), torch.tensor(
        test_data), torch.tensor(train_label), torch.tensor(test_label), torch.tensor(base_mean)

    # Centering and Normalization
    train_data = train_data - base_mean
    train_data = train_data / torch.norm(train_data, dim=1, keepdim=True)
    test_data = test_data - base_mean
    test_data = test_data / torch.norm(test_data, dim=1, keepdim=True)

    # Cross-Domain Shift
    eta = train_data.mean(dim=0, keepdim=True) - test_data.mean(dim=0, keepdim=True)
    test_data = test_data + eta

    inputs = torch.cat([train_data, test_data], dim=0)
    weight = calculate_weight(inputs, args.n_top, args.sigma)
    inputs, train_label, weight = inputs.cuda(), train_label.cuda(), weight.cuda()
    model = DiffusionResNet(n_dim=inputs.shape[1], step_size=args.step_size, layer_num=args.layer_num,
                            weight=weight).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)], gamma=0.1)

    # Prototype Rectification
    whole_data = torch.cat([train_data, test_data], dim=0)
    prototype = train_data.reshape(args.way, args.shot, -1).mean(dim=1)
    cos_sim = F.cosine_similarity(whole_data[:, None, :], prototype[None, :, :], dim=2) * 10  # 10 is a parameter
    pseudo_predict = torch.argmax(cos_sim, dim=1)
    cos_weight = F.softmax(cos_sim, dim=1)
    rectified_prototype = torch.cat(
        [(cos_weight[pseudo_predict == i, i].unsqueeze(1) * whole_data[pseudo_predict == i]).mean(0, keepdim=True)
         for i in range(args.way)], dim=0)
    rectified_prototype = rectified_prototype.cuda()

    for epoch in range(args.epochs):
        train(model, inputs, train_label, optimizer, rectified_prototype)
        scheduler.step()

    outputs = model(inputs)

    # get the accuracy only on query data
    pred = outputs.argmax(dim=1)[args.way * args.shot:].cpu()
    acc = torch.eq(pred, test_label).float().mean().cpu().numpy() * 100
    return acc


def train(model, inputs, train_label, optimizer, prototype):
    outputs = model(inputs)
    loss = nn.CrossEntropyLoss()(outputs[:args.way * args.shot], train_label)

    distance = torch.linalg.norm(inputs[args.way * args.shot:].unsqueeze(1) - prototype.unsqueeze(0), dim=2)
    proto_loss = (F.softmax(outputs[args.way * args.shot:], dim=1) * distance).sum()

    loss = loss + args.alpha * proto_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def simple_shot(embedded_feature, base_mean):
    train_data, test_data, train_label, test_label = sample_task(embedded_feature)

    # Centering and Normalization
    train_data = train_data - base_mean
    train_data = train_data / np.linalg.norm(train_data, axis=1, keepdims=True)
    test_data = test_data - base_mean
    test_data = test_data / np.linalg.norm(test_data, axis=1, keepdims=True)

    prototype = train_data.reshape((args.way, args.shot, -1)).mean(axis=1)
    distance = np.linalg.norm(prototype - test_data[:, None], axis=-1)

    idx = np.argmin(distance, axis=1)
    pred = np.take(np.unique(train_label), idx)
    acc = (pred == test_label).mean() * 100
    return acc


def laplacian_shot(embedded_feature, base_mean, knn=3, lamda=args.lamda, max_iter=20):
    train_data, test_data, train_label, test_label = sample_task(embedded_feature)

    # Centering and Normalization
    train_data = train_data - base_mean
    train_data = train_data / np.linalg.norm(train_data, axis=1, keepdims=True)
    test_data = test_data - base_mean
    test_data = test_data / np.linalg.norm(test_data, axis=1, keepdims=True)

    # Cross-Domain Shift
    eta = train_data.mean(axis=0, keepdims=True) - test_data.mean(axis=0, keepdims=True)
    test_data = test_data + eta

    # Prototype Rectification
    train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)
    whole_data = torch.cat([train_data, test_data], dim=0)
    prototype = train_data.reshape(args.way, args.shot, -1).mean(dim=1)
    cos_sim = F.cosine_similarity(whole_data[:, None, :], prototype[None, :, :], dim=2) * 10  # 10 is a parameter
    pseudo_predict = torch.argmax(cos_sim, dim=1)
    cos_weight = F.softmax(cos_sim, dim=1)
    rectified_prototype = torch.cat(
        [(cos_weight[pseudo_predict == i, i].unsqueeze(1) * whole_data[pseudo_predict == i]).mean(0, keepdim=True)
         for i in range(args.way)], dim=0)

    # calculate weight
    n = test_data.shape[0]
    w = np.zeros((n, n))
    distance = np.linalg.norm(test_data - test_data[:, None], axis=-1)
    knn_ind = np.argsort(distance, axis=1)[:, 1:knn]
    np.put_along_axis(w, knn_ind, 1.0, axis=1)

    # (8a)
    # prototype = train_data.reshape((args.way, args.shot, -1)).mean(axis=1)
    a = np.linalg.norm(rectified_prototype - test_data[:, None], axis=-1)

    y = np.exp(-a) / np.sum(np.exp(-a), axis=1, keepdims=True)
    energy = np.sum(y * (np.log(y) + a - lamda * np.dot(w, y)))

    for i in range(max_iter):
        # (12) update
        out = - a + lamda * np.dot(w, y)
        y = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)

        # (7) check stopping criterion
        energy_new = np.sum(y * (np.log(y) + a - lamda * np.dot(w, y)))
        if abs((energy_new - energy) / energy) < 1e-6:
            break
        energy = energy_new.copy()

    idx = np.argmax(y, axis=1)
    pred = np.take(np.unique(train_label), idx)
    acc = (pred == test_label).mean() * 100
    return acc


if __name__ == '__main__':
    main()
