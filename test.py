import torch
from models.network_onlyattnnoir_kmeans_post import SwinIR

def main():
    net = SwinIR(window_size=8, upsampler='pixelshuffledirect')
    net.cuda()
    
    # x = torch.randn(8, 3, 96, 96).cuda()
    x = torch.randn(4, 3, 64, 64).cuda()
    y = net(x)
    # isnan = y.isnan()
    # print(y)
    # print(isnan.float().sum())
    return

if __name__ == '__main__':
    main()
