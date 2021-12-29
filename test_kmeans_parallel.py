import os
from os import path
from matplotlib import pyplot as plt
import torch
import random
from models import network_onlyattnnoir_kmeans_parallel

def make_gaussian_group(x, y, sigma=0.3):
    px = x + sigma * torch.randn(100)
    py = y + sigma * torch.randn(100)
    g = torch.stack((px, py), dim=-1)
    return g

def main():
    save_dir = 'debug'
    os.makedirs(save_dir, exist_ok=True)
    net = network_onlyattnnoir_kmeans_parallel.SwinIR()
    net.cuda()

    batches = []    
    for b in range(8):
        samples = []
        for n in range(4):
            samples.append(
                make_gaussian_group(
                    random.uniform(-10, 10),
                    random.uniform(-10, 10),
                ),
            )

        # (N, 2)
        samples = torch.cat(samples, dim=0)
        batches.append(samples)

    batches = torch.stack(batches, dim=0)
    batches = batches.cuda()

    c = network_onlyattnnoir_kmeans_parallel.Clustering(k=4)
    centers, labels = c.fit(batches)

    print(centers.size(), labels.size())

    for idx in range(8):
        batch = batches[idx].cpu().numpy()
        label = labels[idx].cpu().numpy()

        plt.figure()
        plt.scatter(batch[:, 0], batch[:, 1], s=1, c=label)
        plt.tight_layout()
        plt.savefig(path.join(save_dir, f'{idx}.png'))
        plt.close()

    return

if __name__ == '__main__':
    main()
