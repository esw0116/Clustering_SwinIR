import torch
# from models.network_onlyattnnoir_iic_blocks import SwinIR
from models.network_onlyattnnoir_gumbel_simple import SwinIR
# from models.network_varswinir2 import SwinIR

def main():
    net = SwinIR(window_size=8, embed_dim=60, mlp_ratio=2., upsampler='pixelshuffledirect', keep_v=True, recycle=True, shifted_window='Full')
    # net = SwinIR(window_size=8, embed_dim=60, mlp_ratio=2., upsampler='pixelshuffledirect',)
    net.cuda()
    net.eval()
    print(net.flops())
    x = torch.randn(6, 3, 96, 96).cuda()
    # y = net(x, print_attn=True)
    y = net(x)

    return

if __name__ == '__main__':
    main()
