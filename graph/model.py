import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLayer(nn.Module):
    def __init__(self, step):
        super(DiffusionLayer, self).__init__()
        self.step = step

    def forward(self, x, adj, diagonal):
        x = x - self.step * torch.matmul(diagonal - adj, x)
        return x


class DiffusionNet(nn.Module):
    def __init__(self, n_features, num_classes, step, layer_num, dropout, diagonal):
        super(DiffusionNet, self).__init__()

        self.linear = nn.Linear(n_features, n_features)

        self.diffusion_layer = DiffusionLayer(step)
        self.classifier = nn.Linear(n_features, num_classes)
        self.dropout = dropout
        self.layer_num = layer_num
        self.diagonal = diagonal

    def forward(self, x, adj):
        x = x + F.relu(self.linear(x))
        for j in range(self.layer_num):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.diffusion_layer(x, adj, self.diagonal)

        out = self.classifier(x)
        return out
