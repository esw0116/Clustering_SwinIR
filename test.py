import torch
from models import network_onlyattnnoir_kmeans

def main():
    net = network_onlyattnnoir_kmeans.SwinIR()
    net.cuda()
    
    x = torch.randn(4, 3, 56, 56).cuda()
    y = net(x)
    # isnan = y.isnan()
    # print(y)
    # print(isnan.float().sum())
    return

if __name__ == '__main__':
    main()
