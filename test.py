import torch
from models import network_onlyattnnoir_kmeans_parallel

def main():
    net = network_onlyattnnoir_kmeans_parallel.SwinIR()
    net.cuda()
    
    x = torch.randn(8, 3, 96, 96).cuda()
    y = net(x)
    #print(y)
    #print(isnan.float().sum())
    return

if __name__ == '__main__':
    main()
