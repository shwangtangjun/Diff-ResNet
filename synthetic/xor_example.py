import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def generate_points(samples_each=100):
    """
    Uniformly sample 100 points from 4 circles in R^2.
    Circles are centered at (0,0),(2,2),(2,0),(0,2), respectively. Their diameters are all 1.
    Points from circles centered at (0,0) and (2,2) belong to class 1. Others belong to class 2.
    :return: Two numpy array of size (100,)
    """
    diameter = 1.5
    radius = diameter / 2
    samples_total = 4 * samples_each
    # Why np.sqrt()? https://stats.stackexchange.com/questions/120527/simulate-a-uniform-distribution-on-a-disc
    r = np.sqrt(np.random.uniform(0, radius ** 2, samples_total))
    theta = np.pi * np.random.uniform(0, 2, samples_total)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    for i in range(samples_each, 3 * samples_each):
        x[i] += 2.
    for i in range(samples_each, 2 * samples_each):
        y[i] += 2.
    for i in range(3 * samples_each, 4 * samples_each):
        y[i] += 2.

    return x, y


def calculate_weight(x, y, sigma=0.5, n_top=20, samples_total=400):
    weight = np.zeros([samples_total, samples_total])
    for i in range(samples_total):
        for j in range(samples_total):
            weight[i, j] = np.exp(-((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2) / sigma ** 2)

    # Sparse and Normalize
    for i in range(samples_total):
        idx = np.argpartition(weight[i], -n_top)[:-n_top]
        weight[i, idx] = 0.
        weight[i] /= weight[i].sum()

    return weight


# naive un-vectorized implementation of diffusion
def diffusion(x, y, weight, step_size=1.0, samples_total=400):
    new_x = np.zeros_like(x)
    new_y = np.zeros_like(y)
    for i in range(samples_total):
        delta_x = 0.
        delta_y = 0.
        for j in range(samples_total):
            delta_x += weight[i, j] * (x[i] - x[j])
            delta_y += weight[i, j] * (y[i] - y[j])
        new_x[i] = x[i] - step_size * delta_x
        new_y[i] = y[i] - step_size * delta_y
    return new_x, new_y


def calculate_l(x, y, samples_total=400):
    min_l = 999.
    samples_each = samples_total / 4
    for i in range(samples_total):
        for j in range(samples_total):
            if i // samples_each != j // samples_each:
                l = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                if l < min_l:
                    min_l = l
    return min_l


def calculate_d(x, y, samples_total=400):
    max_d = 0.
    samples_each = samples_total / 4
    for i in range(samples_total):
        for j in range(samples_total):
            if i // samples_each == j // samples_each:
                d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
                if d > max_d:
                    max_d = d
    return max_d


def main():
    x, y = generate_points()
    weight = calculate_weight(x, y)

    epochs = 201
    for i in range(epochs):
        plt.cla()
        color = [i for i in ['red', 'blue'] for _ in range(200)]

        plt.xticks([])
        plt.yticks([])
        plt.scatter(x, y, c=color, marker='.', animated=True)
        plt.savefig("figures/xor/iter=" + str(i) + ".png", bbox_inches='tight')

        # l = calculate_l(x, y)
        # d = calculate_d(x, y)

        x, y = diffusion(x, y, weight)


if __name__ == '__main__':
    main()
