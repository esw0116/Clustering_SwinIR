import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests

from utils import utils_image as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='swin_sr, real_sr')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth')
    parser.add_argument('--benchmark', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    if os.path.exists(args.model_path):
        # print(f'loading model from {args.model_path}')
        pass
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)
        
    model = define_model(args)
    model.eval()
    model = model.to(device)

    if args.benchmark == 'Set5':
        args.folder_lq = 'dataset/benchmark/Set5/LR_bicubic/X{}'.format(args.scale)
        args.folder_gt = 'dataset/benchmark/Set5/HR'
    elif args.benchmark == 'Urban100':
        args.folder_lq = 'dataset/benchmark/Urban100/LR_bicubic/X{}'.format(args.scale)
        args.folder_gt = 'dataset/benchmark/Urban100/HR'
    
    # setup folder and path
    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['latency'] = []
    psnr, ssim, psnr_y, ssim_y, latency = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model, args, window_size)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
        end.record()
        torch.cuda.synchronize()
        runtime = start.elapsed_time(end)
        test_results['latency'].append(runtime)
        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)
            # print(img_gt.shape, output.shape)
            psnr = util.calculate_psnr(output, img_gt, border=border, y_psnr=False)
            ssim = 0
            # ssim = util.calculate_ssim(output, img_gt, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                output_y = util.bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border, y_psnr=False)
                ssim_y = 0
                # ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
            # print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
            #       'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
            #       'Latency: {:.2f} sec.'.
            #       format(idx, imgname, psnr, ssim, psnr_y, ssim_y, runtime))
        else:
            print('Testing {:d} {:20s}'.format(idx, imgname))

    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))
        ave_runtime = sum(test_results['latency'][1:]) / len(test_results['latency'][1:])
        print('-- Average Runtime: {:.2f} sec'.format(ave_runtime))

def define_model(args):
    # use 'pixelshuffledirect' to save parameters
    if args.task == 'swin_sr':
        from models.network_swinir import SwinIR as net
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'
    
    elif args.task == 'noswin_sr':
        from models.network_noswinir import SwinIR as net
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task in ['onlySA_sr', 'onlySA_shallow_sr', 'onlySA_tooshallow_sr', 'onlySA_short_sr', 'onlySA_tooshort_sr', 'onlySA_thin_sr', 'onlySA_toothin_sr']:
        from models.network_onlyattnir import SwinIR as net
        if args.task == 'onlySA_shallow_sr':
            dim = 48
        elif args.task == 'onlySA_tooshallow_sr':
            dim = 42
        else:
            dim=60
        
        if args.task == 'onlySA_short_sr':
            depth = [6,6,6]
            head = [6,6,6]
        elif args.task == 'onlySA_tooshort_sr':
            depth = [6,6]
            head = [6,6]
        elif args.task == 'onlySA_thin_sr':
            depth = [4,4,4,4]
            head = [6,6,6,6]
        elif args.task == 'onlySA_toothin_sr':
            depth = [3,3,3,3]
            head = [6,6,6,6]
        else:
            depth = [6,6,6,6]
            head = [6,6,6,6]
        
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=depth, embed_dim=dim, num_heads=head,
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task in ['kmeans_sr', 'kmeans_post_sr', 'kmeans_postkeepv_sr', 'kmeans_postkeepvnorecycle_sr', 'kmeans_last_sr']:
        from models.network_onlyattnnoir_kmeans_blocks import SwinIR as net
        
        if args.task == 'kmeans_sr':
            block = ['RTB','RPCTB','RTB','RPCTB']
        elif args.task == 'kmeans_last_sr':
            block=['RPCTB','RPCTB','RPCTB','RTB']
        else:
            block=['RPCTB','RTB','RPCTB','RTB']
            
        if args.task == 'kmeans_postkeepv_sr' or args.task == 'kmeans_postkeepvnorecycle_sr':
            keepv = True
        else:
            keepv = False

        if args.task == 'kmeans_postkeepvnorecycle_sr':
            recycle = False
        else:
            recycle = True
        
        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8,
                 img_range=1., embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 blocks=block, num_groups=16, keep_v=keepv, recycle=recycle,
                 mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'
    
    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    # if args.task == 'swin_sr':
    hr = os.path.basename(os.path.dirname(args.folder_gt))
    save_dir = f'results/{hr}/{args.task}_X{args.scale}'
    folder = args.folder_gt
    border = args.scale
    window_size = 8

    # elif args.task == 'noswin_sr':
    #     hr = os.path.basename(os.path.dirname(args.folder_gt))
    #     save_dir = f'results/{hr}/{args.task}_X{args.scale}'
    #     folder = args.folder_gt
    #     border = args.scale
    #     window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    # if args.task in ['swin_sr']:
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
