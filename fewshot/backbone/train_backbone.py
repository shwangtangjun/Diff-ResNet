import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import network
import numpy as np
import collections
from backbone_utils import get_configuration, get_train_dataloader, get_tqdm, get_val_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int, help='seed for training')
parser.add_argument("--dataset", choices=['mini', 'tiered', 'cub'], type=str)
parser.add_argument("--backbone", choices=['resnet18', 'wideres'], type=str, help='network architecture')
parser.add_argument('--epochs', type=int, help='number of training epochs. 100 for mini and tiered. 400 for cub')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--silent', action='store_true', help='call --silent to disable tqdm')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # specify which GPU(s) to be used


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    data_path, split_path, save_path, num_classes = get_configuration(args.dataset, args.backbone)
    train_loader = get_train_dataloader(data_path, split_path, args.batch_size)
    val_loader = get_val_dataloader(data_path, split_path)

    model = network.__dict__[args.backbone](num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)], gamma=0.1)

    tqdm_epochs = get_tqdm(range(args.epochs), args.silent)
    if not args.silent:
        tqdm_epochs.set_description('Total Epochs')

    if not os.path.isdir('../saved_models'):
        os.makedirs('../saved_models')

    best_acc = 0
    for epoch in tqdm_epochs:
        train(train_loader, model, optimizer, epoch)
        scheduler.step()

        if epoch >= int(.75 * args.epochs):
            val_acc = validate(val_loader, model)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)


def train(train_loader, model, optimizer, epoch):
    model.train()

    correct_count = 0
    total_count = 0
    acc = 0
    tqdm_train_loader = get_tqdm(train_loader, args.silent)

    for batch_idx, (inputs, labels) in enumerate(tqdm_train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1)
        correct_count += pred.eq(labels).sum().item()
        total_count += len(inputs)
        acc = correct_count / total_count * 100

        if not args.silent:
            tqdm_train_loader.set_description('Acc {:.2f}'.format(acc))

    if args.silent:
        print("Epoch={}, Accuracy={:.2f}".format(epoch + 1, acc))


# Below codes only used for validation. We save the models with the highest 1-shot nearest neighbor classification
# accuracy.
def validate(val_loader, model):
    input_dict = collections.defaultdict(list)
    for i, (inputs, labels) in enumerate(val_loader):
        for img, label in zip(inputs, labels):
            input_dict[label.item()].append(img)

    acc_list = []
    tqdm_test_iter = get_tqdm(range(1000), args.silent)
    for _ in tqdm_test_iter:
        acc = nearest_prototype(input_dict, model)
        acc_list.append(acc)

        if not args.silent:
            tqdm_test_iter.set_description('Validate on few-shot tasks. Accuracy:{:.2f}'.format(np.mean(acc_list)))
    if args.silent:
        print("Validation Accuracy={:.2f}".format(np.mean(acc_list)))

    return np.mean(acc_list)


def nearest_prototype(input_dict, model):
    sample_class = random.sample(list(input_dict.keys()), 5)
    train_img, test_img, test_label, train_label = [], [], [], []
    for i, each_class in enumerate(sample_class):
        samples = random.sample(input_dict[each_class], 1 + 15)

        train_label += [i] * 1  # We only validate on 1-shot tasks, for simplicity
        test_label += [i] * 15
        train_img += samples[:1]
        test_img += samples[1:]

    train_img, test_img = torch.stack(train_img).cuda(), torch.stack(test_img).cuda()
    train_test_img = torch.cat([train_img, test_img])

    train_label, test_label = np.array(train_label), np.array(test_label)

    model.eval()
    with torch.no_grad():
        train_test_data, _ = model(train_test_img, return_feature=True)

    train_test_data = train_test_data.cpu().data.numpy()
    train_data, test_data = train_test_data[:5], train_test_data[5:]

    prototype = train_data.reshape((5, 1, -1)).mean(axis=1)
    distance = np.linalg.norm(prototype - test_data[:, None], axis=-1)

    idx = np.argmin(distance, axis=1)
    pred = np.take(np.unique(train_label), idx)
    acc = (pred == test_label).mean() * 100
    return acc


if __name__ == '__main__':
    main()
