import os
import collections
import pickle
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
import backbone.network as network
from tqdm import tqdm
import PIL.Image as Image
import re


def get_tqdm(iters, silent):
    """
    Wrap iters with tqdm if not --silent
    """
    if silent:
        return iters
    else:
        return tqdm(iters)


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    """
    return np.mean(data), 1.96 * (np.std(data) / np.sqrt(len(data)))


def calculate_weight(inputs, n_top, sigma):
    distance = torch.norm(inputs.unsqueeze(0) - inputs.unsqueeze(1), dim=-1)
    dist_n_top = torch.kthvalue(distance, n_top, dim=1, keepdim=True)[0]
    dist_sigma = torch.kthvalue(distance, sigma, dim=1, keepdim=True)[0]

    distance_truncated = distance.where(distance < dist_n_top, torch.tensor(float("inf")))
    weight = torch.exp(-(distance_truncated / dist_sigma).pow(2))

    # Symmetrically normalize the weight matrix
    d_inv_sqrt = torch.diag(weight.sum(dim=1).pow(-0.5))
    weight = d_inv_sqrt.mm(weight).mm(d_inv_sqrt)
    weight = (weight + weight.t()) / 2
    weight = weight.detach()
    return weight


class DatasetFolder(data.Dataset):
    def __init__(self, root, split_dir, split_type, transform):
        assert split_type in ['train', 'val', 'test']
        split_file = os.path.join(split_dir, split_type + '.csv')
        assert os.path.isfile(split_file)

        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']

        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]

        self.root = root
        self.transform = transform
        self.data = data
        self.labels = mapped_labels
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        filename = self.data[index]
        path_file = os.path.join(self.root, filename)
        assert os.path.isfile(path_file)
        img = Image.open(path_file).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloader(data_path, split_path, split_type):
    dataset = re.split('[/_]', data_path)[-2]
    # First resize larger than 84, then center crop, achieve better result
    if dataset == "cub":
        resize = 120
    else:
        resize = 96
    datasets = DatasetFolder(root=data_path, split_dir=split_path, split_type=split_type,
                             transform=transforms.Compose([transforms.Resize(resize),
                                                           transforms.CenterCrop(84),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])]))

    loader = torch.utils.data.DataLoader(datasets, batch_size=1000, shuffle=False, num_workers=40)
    return loader


def get_embedded_feature(test_loader, save_path, silent):
    """
    Return embedded features of data from novel classes
    """
    # Only compute once for each dataset+backbone
    if os.path.isfile(save_path + '_embedded_feature.plk'):
        embedded_feature = load_pickle(save_path + '_embedded_feature.plk')
        return embedded_feature

    model = load_pretrained_backbone(save_path)

    model.eval()
    with torch.no_grad():
        embedded_feature = collections.defaultdict(list)

        tqdm_test_loader = get_tqdm(test_loader, silent)
        if not silent:
            tqdm_test_loader.set_description('Computing embedded features on test classes')

        for i, (inputs, labels) in enumerate(tqdm_test_loader):
            features, _ = model(inputs, return_feature=True)
            features = features.cpu().data.numpy()
            for feature, label in zip(features, labels):
                embedded_feature[label.item()].append(feature)
        save_pickle(save_path + '_embedded_feature.plk', embedded_feature)

    return embedded_feature


def get_base_mean(train_loader, save_path, silent):
    """
    Return average of data from base classes
    """
    # Only compute once for each dataset+backbone
    if os.path.isfile(save_path + '_base_mean.plk'):
        base_mean = load_pickle(save_path + '_base_mean.plk')
        return base_mean

    model = load_pretrained_backbone(save_path)

    model.eval()
    with torch.no_grad():
        base_mean = []

        tqdm_train_loader = get_tqdm(train_loader, silent)
        if not silent:
            tqdm_train_loader.set_description('Computing average on base classes')

        for i, (inputs, _) in enumerate(tqdm_train_loader):
            outputs, _ = model(inputs, return_feature=True)
            outputs = outputs.cpu().data.numpy()
            base_mean.append(outputs)
        base_mean = np.concatenate(base_mean, axis=0).mean(axis=0)
        save_pickle(save_path + '_base_mean.plk', base_mean)
    return base_mean


def get_configuration(dataset, backbone):
    """
    Get configuration according to dataset and backbone.
    """

    data_path = './data/' + dataset + '/images'
    split_path = './data/' + dataset + '/split'
    save_path = './saved_models/' + dataset + '_' + backbone

    if dataset == 'mini':
        num_classes = 64
    elif dataset == 'tiered':
        num_classes = 351
    elif dataset == 'cub':
        num_classes = 100
    else:
        raise NotImplementedError

    return data_path, split_path, save_path, num_classes


def load_pretrained_backbone(save_path):
    dataset = re.split('[/_]', save_path)[-2]
    backbone = re.split('[/_]', save_path)[-1]

    if dataset == 'mini':
        num_classes = 64
    elif dataset == 'tiered':
        num_classes = 351
    elif dataset == 'cub':
        num_classes = 100
    else:
        raise NotImplementedError

    model = network.__dict__[backbone](num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(save_path + '.pt'))

    return model
