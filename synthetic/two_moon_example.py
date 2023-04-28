import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

random.seed(40)
np.random.seed(40)
torch.manual_seed(40)


def make_moons(n_samples=500):
    """Make two interleaving half circles.
    A simple toy dataset to visualize clustering and classification
    algorithms. Read more in the :ref:`User Guide <sample_generators>`.
    Parameters
    """
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples))
    inner_circ_y = 0.5 - np.sin(np.linspace(0, np.pi, n_samples))

    x = np.append(outer_circ_x, inner_circ_x)
    y = np.append(outer_circ_y, inner_circ_y)

    x += np.random.randn(1000) * 0.05
    y += np.random.randn(1000) * 0.05
    return x, y


def calculate_weight(x, y, sigma=0.5, n_top=25):
    weight = np.zeros([1000, 1000])
    for i in range(1000):
        for j in range(1000):
            weight[i, j] = np.exp(-((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) / sigma ** 2)

    # Sparse and Normalize
    for i in range(1000):
        idx = np.argpartition(weight[i], -n_top)[:-n_top]
        weight[i, idx] = 0.
        weight[i] /= weight[i].sum()

    return weight


class DiffusionLayer(nn.Module):
    def __init__(self, step):
        super(DiffusionLayer, self).__init__()
        self.step = step

    def forward(self, x, adj):
        diagonal = torch.eye(x.size(0), device=x.device)
        x = x - self.step * torch.matmul(diagonal - adj, x.flatten(1)).view_as(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 2)
        self.layer_num = 60
        self.diffusion_layer = DiffusionLayer(step=1.0)

    def forward(self, x, weight):
        out = self.fc2(F.relu(self.fc1(x))) + x

        # Uncomment following lines to use diffusion
        # for i in range(self.layer_num):
        #     out = self.diffusion_layer(out, weight)
        res = self.classifier(out)
        return res, out


def train(model, inputs, weight, labels):
    optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9, weight_decay=5e-4)
    optimizer.zero_grad()

    outputs, features = model(inputs, weight)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    optimizer.step()


def test(model, inputs, weight, labels):
    outputs, features = model(inputs, weight)

    pred = outputs.argmax(1)
    acc = torch.eq(pred, labels).sum()
    return acc.item()


def main():
    x, y = make_moons()

    color = [i for i in ['red', 'blue'] for _ in range(500)]
    plt.scatter(x, y, c=color, marker='.')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("figures/two_moon/raw.png", bbox_inches='tight')

    weight = calculate_weight(x, y)
    x, y, weight = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weight)
    inputs = torch.stack([x, y], dim=1).float()
    weight = weight.float()
    labels = torch.cat([torch.zeros(500), torch.ones(500)]).long()

    acc_list = np.zeros(21)
    model = Net()
    for epoch in range(21):
        outputs, features = model(inputs, weight)
        x, y = features[:, 0].detach().numpy(), features[:, 1].detach().numpy()
        acc = test(model, inputs, weight, labels)
        print(epoch, acc)
        acc_list[epoch] = acc

        plt.cla()
        color = [i for i in ['red', 'blue'] for _ in range(500)]
        plt.scatter(x, y, c=color, marker='.')
        plt.title("accuracy=" + str(round(acc / 1000 * 100, 1)) + "%", fontsize=40)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("figures/two_moon/without_diffusion_iter=" + str(epoch) + ".png", bbox_inches='tight')

        train(model, inputs, weight, labels)
    print(acc_list)


if __name__ == '__main__':
    main()
