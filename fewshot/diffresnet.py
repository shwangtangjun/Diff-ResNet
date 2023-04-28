import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLayer(nn.Module):
    def __init__(self, step_size, laplacian):
        super(DiffusionLayer, self).__init__()
        self.step_size = step_size
        self.laplacian = laplacian

    def forward(self, x):
        x = x - self.step_size * torch.matmul(self.laplacian, x.flatten(1)).view_as(x)
        return x


class DiffusionResNet(nn.Module):
    def __init__(self, n_dim, step_size, layer_num, weight):
        super(DiffusionResNet, self).__init__()
        self.layer_num = layer_num
        diagonal = torch.diag(weight.sum(dim=1))
        laplacian = diagonal - weight

        self.fc1 = nn.Linear(n_dim, n_dim)
        self.fc2 = nn.Linear(n_dim, n_dim)
        self.classifier = nn.Linear(n_dim, 5)  # 5-way classification
        self.diffusion_layer = DiffusionLayer(step_size, laplacian)

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x))) + x
        for _ in range(self.layer_num):
            x = self.diffusion_layer(x)
        out = self.classifier(x)
        return out
