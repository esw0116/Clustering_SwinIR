import torch
from models.network_swinir import SwinIR

def main():
    net = SwinIR(window_size=8, upsampler='pixelshuffledirect')
    net.cuda()
    
    x = torch.randn(4, 3, 64, 64).cuda()
    y = net(x, print_attn=True)

    return

if __name__ == '__main__':
    main()
