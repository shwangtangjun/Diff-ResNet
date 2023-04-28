import os
import numpy as np
import PIL.Image as Image
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
import torch
import re


def get_configuration(dataset, backbone):
    """
    Get configuration according to dataset and backbone.
    """

    data_path = '../data/' + dataset + '/images'
    split_path = '../data/' + dataset + '/split'
    save_path = '../saved_models/' + dataset + '_' + backbone + '.pt'

    if dataset == 'mini':
        num_classes = 64
    elif dataset == 'tiered':
        num_classes = 351
    elif dataset == 'cub':
        num_classes = 100
    else:
        raise NotImplementedError

    return data_path, split_path, save_path, num_classes


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


def get_train_dataloader(data_path, split_path, batch_size):
    datasets = DatasetFolder(root=data_path, split_dir=split_path, split_type='train',
                             transform=transforms.Compose([transforms.RandomResizedCrop(84),
                                                           transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                                                                  saturation=0.4),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])]))

    # Setting appropriate num_workers can significantly increase training speed
    loader = data.DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=40, pin_memory=True)

    return loader


def get_val_dataloader(data_path, split_path):
    dataset = re.split('[/_]', data_path)[-2]
    if dataset == "cub":
        resize = 120
    else:
        resize = 96
    datasets = DatasetFolder(root=data_path, split_dir=split_path, split_type='val',
                             transform=transforms.Compose([transforms.Resize(resize),
                                                           transforms.CenterCrop(84),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])]))
    loader = torch.utils.data.DataLoader(datasets, batch_size=100, shuffle=False, num_workers=40)
    return loader


def get_tqdm(iters, silent):
    """
    Wrap iters with tqdm if not --silent
    """
    if silent:
        return iters
    else:
        return tqdm(iters)
