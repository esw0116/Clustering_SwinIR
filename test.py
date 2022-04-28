import torch
from models.network_onlyattnnoir_gumbel_simple import SwinIR

def main():
    net = SwinIR(window_size=8, embed_dim=60, mlp_ratio=2., upsampler='pixelshuffledirect', keep_v=True, recycle=True, shifted_window='Half', blocks=['RPCTB','RPCTB', 'RPCTB','RPCTB'])
    net.cuda()
    net.eval()
    print(net.flops())
    x = torch.randn(6, 3, 96, 96).cuda()
    # y = net(x, print_attn=True)
    y = net(x)

    return

if __name__ == '__main__':
    main()
