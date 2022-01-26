import torch
# from models.network_swinir import SwinIR
# from models.network_blockcompnoswinir2 import SwinIR
from models.network_onlyattnnoir_halfkmeans_blocks import SwinIR

def main():
    net = SwinIR(window_size=8, upsampler='pixelshuffledirect')
    net.cuda()
    
    x = torch.randn(6, 3, 96, 96).cuda()
    # y = net(x, print_attn=True)
    y = net(x)

    return

if __name__ == '__main__':
    main()
