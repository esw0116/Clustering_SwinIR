import cv2
import imageio
import numpy as np
import torch
from models.network_onlyattnir import SwinIR as net

def main():
    model = net(upscale=2, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    param_key_g = 'params'
    pretrained_model = torch.load('nsmltrained_models/KR80934_CVLAB_SR_200/250000/model/G.pth')
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model = model.cuda()
    
    scale = 2
    ws = 8
    x = 120
    y = 228
    
    imgfile = 'dataset/benchmark/Set5/LR_bicubic/X{}/babyx{}.png'.format(scale, scale)
    img = cv2.imread(imgfile, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = np.transpose(img if img.shape[2] == 1 else img[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    img = torch.from_numpy(img).float().unsqueeze(0).cuda()  # CHW-RGB to NCHW-RGB
    
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        img_part = img[:,:,y:y+ws,x:x+ws]
        imageio.imwrite('results/attnmap/imgpart.png', img_part.permute(0,2,3,1).squeeze(0).cpu().numpy())
        # output = test(img_lq, model, args, window_size, imgname=f'{save_dir}/{imgname}')
        y = model(img_part, print_attn=True)
    return

if __name__ == '__main__':
    main()
