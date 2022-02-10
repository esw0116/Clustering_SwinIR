import torch
from models.network_onlyattnnoir_gumbel_blocks import SwinIR

def main():
    net = SwinIR(window_size=8, upsampler='pixelshuffledirect', keep_v=True)
    net.cuda()
    net.eval()
    x = torch.randn(6, 3, 96, 96).cuda()
    # y = net(x, print_attn=True)
    y = net(x)

    return

if __name__ == '__main__':
    main()
