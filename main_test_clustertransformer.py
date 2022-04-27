import argparse
import cv2
import os, glob
import numpy as np
from collections import OrderedDict
import torch
import math, time
from thop import profile, clever_format

from utils import utils_image as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='color_dn', help='swin_sr, real_sr')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--benchmark', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--time', action='store_true')
    args = parser.parse_args()

    # device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    device = torch.device('cuda' if not args.cpu  else 'cpu')
    
    model = define_model(args)
    model.eval()
    model = model.to(device)

    if args.benchmark == 'Set5':
        args.folder_lq = 'dataset/benchmark/Set5/LR_bicubic/X{}'.format(args.scale)
        args.folder_gt = 'dataset/benchmark/Set5/HR'
    if args.benchmark == 'Set14':
        args.folder_lq = 'dataset/benchmark/Set14/LR_bicubic/X{}'.format(args.scale)
        args.folder_gt = 'dataset/benchmark/Set14/HR'
    if args.benchmark == 'B100':
        args.folder_lq = 'dataset/benchmark/B100/LR_bicubic/X{}'.format(args.scale)
        args.folder_gt = 'dataset/benchmark/B100/HR'
    elif args.benchmark == 'Urban100':
        args.folder_lq = 'dataset/benchmark/Urban100/LR_bicubic/X{}'.format(args.scale)
        args.folder_gt = 'dataset/benchmark/Urban100/HR'
    elif args.benchmark == 'div2k':
        args.folder_lq = 'dataset/DIV2K/DIV2K_valid_LR_bicubic/X{}'.format(args.scale)
        args.folder_gt = 'dataset/DIV2K/DIV2K_valid_HR'

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

    if not args.cpu:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB
        if args.bs > 1:
            img_lq = img_lq.expand((args.bs, -1, -1, -1))
        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = math.ceil(h_old / (2*window_size)) * 2*window_size - h_old
            w_pad = math.ceil(w_old / (2*window_size)) * 2*window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            # output = test(img_lq, model, args, window_size, imgname=f'{save_dir}/{imgname}')
            if args.benchmark == 'div2k':
                if args.scale == 2:
                    img_lq = img_lq[..., :368, :640]
                elif args.scale == 3:
                    img_lq = img_lq[..., :240, :416]
                elif args.scale == 4:
                    img_lq = img_lq[..., :192, :320]
            
            if args.cpu:
                start = time.time()
            else:
                start.record()
            output = test(img_lq, model, args, window_size, detail_time=args.time)
            if args.cpu:
                end = time.time()
                runtime = (end - start) * 1000
            else:
                end.record()
                torch.cuda.synchronize()
                runtime = start.elapsed_time(end)

            if args.benchmark == 'div2k' and idx == 0:
                macs, params = profile(model, inputs=(img_lq, ))
                macs, params = clever_format([macs, params], "%.3f")
                print(params, macs)
            output = output[..., :h_old * args.scale, :w_old * args.scale]
            test_results['latency'].append(runtime)
            if args.time:
                input()
        
        # save image
        output = output[0:1]
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        # cv2.imwrite(f'{save_dir}/{imgname}_SwinIR.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)
            # print(img_gt.shape, output.shape)
            psnr = 0  # util.calculate_psnr(output, img_gt, border=border, y_psnr=False)
            ssim = 0
            # ssim = util.calculate_ssim(output, img_gt, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                output_y = util.bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                if args.benchmark == 'div2k':
                    psnr_y = 0 #util.calculate_psnr(output_y, img_gt_y, border=border, y_psnr=False)
                    ssim_y = 0
                else:
                    psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border, y_psnr=False)
                    ssim_y = util.calculate_ssim_pth(output_y, img_gt_y, border=border)

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
        print('\n{} \n-- Params number: {} \n-- Flops: {} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(
            save_dir, sum(map(lambda x: x.numel(), model.parameters())), model.flops(), ave_psnr, ave_ssim))

        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))
        ave_runtime = sum(test_results['latency'][1:]) / len(test_results['latency'][1:])
        # print(len(test_results['latency'][1:]))
        print('-- Average Runtime: {:.2f} sec'.format(ave_runtime))

def define_model(args):
    # use 'pixelshuffledirect' to save parameters
    if args.task == 'swin_sr':
        if args.cpu:
            from models.network_swinir_cpu import SwinIR as net
        else:
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

    elif args.task in ['kmeans_sr', 'kmeans_post_sr', 'kmeans_postkeepv_sr', 'kmeans_postkeepvnorecycle_sr', 'kmeans_last_sr', 'kmeans_postkeepv_bias_sr',
                        'kmeans_tooshort_sr', 'kmeans_tooshortkeepv_sr', 'kmeans_postkeepv_ws_sr', 
                        'gumbel_postkeepv_ws_sr', 'gumbel_postkeepvnorecycle_ws_sr', 'gumbel_postkeepv_sr', 'iic_postkeepv_ws_sr']:
        if args.task.startswith('kmeans'):
            from models.network_onlyattnnoir_kmeans_blocks import SwinIR as net
        elif args.task.startswith('gumbel'):
            from models.network_onlyattnnoir_gumbel_blocks import SwinIR as net
        elif args.task.startswith('iic'):
            from models.network_onlyattnnoir_iic_blocks import SwinIR as net
        depth = [6,6,6,6]
        head = [6,6,6,6]
        if 'post' in args.task:
            block = ['RPCTB','RTB','RPCTB','RTB']
        elif 'last' in args.task:
            block=['RPCTB','RPCTB','RPCTB','RTB']
        elif 'tooshort' in args.task:
            block=['RPCTB','RPCTB']
            depth = [6,6]
            head = [6,6]
        else:
            block=['RTB','RPCTB','RTB','RPCTB']
            
        if 'keepv' in args.task:
            keepv = True
        else:
            keepv = False
        
        if '_ws' in args.task:
            shift_window = True
        else:
            shift_window = False

        if 'norecycle' in args.task:
            recycle = False
        else:
            recycle = True
        
        if '_bias' in args.task:
            relative_bias = True
        else:
            relative_bias = False

        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8,
                 img_range=1., embed_dim=60, depths=depth, num_heads=head,
                 blocks=block, num_groups=16, keep_v=keepv, recycle=recycle, relative_bias=relative_bias, shifted_window=shift_window,
                 mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'
    
    elif args.task in ['kmeans_final_small_yes', 'kmeans_final_small_no', 'kmeans_final_big_yes', 'kmeans_final_big_no',
                        'gumbel_final_small_yes', 'gumbel_final_small_no', 'gumbel_final_big_yes', 'gumbel_final_big_no',]:
        if args.task.startswith('kmeans'):
            from models.network_onlyattnnoir_kmeans_final import SwinIR as net
        elif args.task.startswith('gumbel'):
            if args.cpu:
                from models.network_onlyattnnoir_gumbel_final_cpu import SwinIR as net
            else:
                from models.network_onlyattnnoir_gumbel_final import SwinIR as net


        if 'small' in args.task:
            block = ['RPCTB','RTB','RPCTB','RTB']
            depth = [6,6,6,6]
            head = [6,6,6,6]
            embed_dim = 60
        elif 'big' in args.task:
            block = ['RPCTB','RPCTB','RTB','RPCTB','RPCTB','RTB']
            depth = [6,6,6,6,6,6]
            head = [6,6,6,6,6,6]
            embed_dim = 180
        else:
            block=['RTB','RPCTB','RTB','RPCTB']
            
        keepv = True
        shift_window = 'Half'

        if args.task.endswith('no'):
            recycle = False
        else:
            recycle = True
        
        if '_bias' in args.task:
            relative_bias = True
        else:
            relative_bias = False

        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8,
                 img_range=1., embed_dim=embed_dim, depths=depth, num_heads=head,
                 blocks=block, num_groups=8, keep_v=keepv, recycle=recycle, relative_bias=relative_bias, shifted_window=shift_window,
                 mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task in ['kmeans_simple_small_yes', 'kmeans_simple_small_no', 'kmeans_simple_big_yes', 'kmeans_simple_big_no',
                        'gumbel_simple_small_yes', 'gumbel_simple_small_no', 'gumbel_simple_big_yes', 'gumbel_simple_big_no', 'gumbel_toosimple_small_yes',]:
        if args.task.startswith('kmeans'):
            from models.network_onlyattnnoir_kmeans_simple import SwinIR as net
        elif args.task.startswith('gumbel'):
            if 'too' in args.task:
                from models.network_onlyattnnoir_gumbel_toosimple import SwinIR as net
            else:
                from models.network_onlyattnnoir_gumbel_simple import SwinIR as net

        if 'small' in args.task:
            block = ['RPCTB','RTB','RPCTB','RTB']
            depth = [6,6,6,6]
            head = [6,6,6,6]
            embed_dim = 60
        elif 'big' in args.task:
            block = ['RPCTB','RPCTB','RTB','RPCTB','RPCTB','RTB']
            depth = [6,6,6,6,6,6]
            head = [6,6,6,6,6,6]
            embed_dim = 180
        else:
            block=['RTB','RPCTB','RTB','RPCTB']
            
        keepv = True
        shift_window = 'Half'

        if args.task.endswith('no'):
            recycle = False
        else:
            recycle = True
        
        if '_bias' in args.task:
            relative_bias = True
        else:
            relative_bias = False

        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8,
                 img_range=1., embed_dim=embed_dim, depths=depth, num_heads=head,
                 blocks=block, num_groups=8, keep_v=keepv, recycle=recycle, relative_bias=relative_bias, shifted_window=shift_window,
                 mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task == 'mixed':
        from models.network_onlyattnnoir_mixed2 import SwinIR as net
        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8, groupwindow_ratio=2,
                        img_range=1., embed_dim=60, depths=[6,6,6,6], num_heads=[6,6,6,6],
                        blocks=['RPCTB','RPCTB', 'RPCTB','RPCTB'], num_groups=8, keep_v=True, recycle=True, relative_bias=False, shifted_window='Half',
                        mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task in ['cascade', 'cascade32', 'cascade64']:
        from models.network_onlyattnnoir_cascade_noLN import SwinIR as net
        if args.task == 'cascade32':
            gwr = 4
            ng = 8
        elif args.task == 'cascade64':
            gwr = 8
            ng = 16
        else:
            gwr = 2
            ng = 8

        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8, groupwindow_ratio=gwr,
                        img_range=1., embed_dim=60, depths=[6,6], num_heads=[6,6],
                        blocks=['RPCTB','RPCTB'], num_groups=ng, keep_v=True, recycle=True, relative_bias=False, shifted_window='Half',
                        mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task == 'noln':
        from models.network_onlyattnnoir_noLN import SwinIR as net
        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8, groupwindow_ratio=2,
                        img_range=1., embed_dim=60, depths=[6,6,6,6], num_heads=[6,6,6,6],
                        blocks=['RPCTB','RPCTB', 'RPCTB','RPCTB'], num_groups=8, keep_v=True, recycle=True, relative_bias=False, shifted_window='Half',
                        mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task == 'kmeans_normpost_sr':
        from models.network_onlyattnnoir_kmeans_normblocks import SwinIR as net
        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8,
                 img_range=1., embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 blocks=['RPCTB','RTB','RPCTB','RTB'], num_groups=16, keep_v=False, recycle=True,
                 mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task == 'intrakmeans_post_sr':
        from models.network_onlyattnnoir_intrakmeans_blocks import SwinIR as net
        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8,
                 img_range=1., embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 blocks=['RPCTB','RTB','RPCTB','RTB'], num_groups=16, keep_v=False, recycle=True,
                 mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task == 'halfkmeans_post_sr':
        from models.network_onlyattnnoir_halfkmeans_blocks import SwinIR as net
        model = net(upscale=args.scale, img_size=64, in_chans=3, window_size=8,
                 img_range=1., embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 blocks=['RPCTB','RTB','RPCTB','RTB'], num_groups=16, keep_v=False, recycle=True,
                 mlp_ratio=2., upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task == 'blockcluster_noswin_sr':
        from models.network_blockcompnoswinir2 import SwinIR as net
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'
    
    elif args.task == 'random_noswin_sr':
        from models.backup.network_onlyattnnoir_random_blocks import SwinIR as net
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    elif args.task == 'randomfix_noswin_sr':
        from models.backup.network_onlyattnnoir_randomfix_blocks import SwinIR as net
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    if not args.model_path == '':
        pretrained_model = torch.load(args.model_path)
        model_params = pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model
        valid_model_params = {}
        for k, v in model_params.items():
            if k.endswith('attn_mask'):
                pass
            else:
                valid_model_params[k] = v

        model.load_state_dict(valid_model_params, strict=False)
    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    # if args.task == 'swin_sr':
    hr = os.path.basename(os.path.dirname(args.folder_gt))
    save_dir = f'results/{hr}/{args.task}_X{args.scale}'
    folder = args.folder_gt
    border = args.scale
    # border = 0
    window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
    # if args.task in ['swin_sr']:
    img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
        np.float32) / 255.

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size, imgname=None, detail_time=True):
    if args.tile is None:
        # test the image as a whole
        if imgname is None:
            output = model(img_lq, print_time=detail_time)
        else:
            output = model(img_lq, imgsave_name=imgname)
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
