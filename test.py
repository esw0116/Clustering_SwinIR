import torch
from models import network_onlyattnnoir_kmeans

def main():
    net = network_onlyattnnoir_kmeans.SwinIR(window_size=8, upsampler='pixelshuffledirect')
    net.cuda()
    
    x = torch.randn(8, 3, 96, 96).cuda()
    y = net(x)
    #print(y)
    #print(isnan.float().sum())
    return

if __name__ == '__main__':
    main()
