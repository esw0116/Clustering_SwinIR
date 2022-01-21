import cv2
import imageio
import numpy as np
import os
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
        img_part = img[:,:,y-ws:y+2*ws,x-ws:x+2*ws]
        imageio.imwrite('results/attnmap/imgpart.png', img_part.permute(0,2,3,1).squeeze(0).cpu().numpy())
        y, attnmap_list = model(img_part, print_attn=True)
        for i, attnmap_sublist in enumerate(attnmap_list):
            for j, attnmap in enumerate(attnmap_sublist):
                B, h, w, n = attnmap.shape
                assert B == (img_part.shape[-2] // ws) * (img_part.shape[-1] // ws)
                attnmap_max = np.amax(attnmap, axis=2, keepdims=True)
                attnmap = attnmap / attnmap_max
                for k in range(n):
                    attnmap_head = attnmap[..., k]
                    save_folder = 'results/attnmap/map/{}'.format(111+100*i+10*j+k)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    for q in range(h):
                        visual_map = attnmap_head[:, q]
                        visual_map = visual_map.reshape(B, ws, ws).reshape(ws*(img_part.shape[-2] // ws), ws*(img_part.shape[-1] // ws))
                        visual_map = (visual_map * 255).astype('uint8')
                        imageio.imwrite(os.path.join(save_folder, 'pixel_{:02d}.png'.format(q)), visual_map)

if __name__ == '__main__':
    main()
