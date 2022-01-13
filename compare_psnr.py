import argparse
import glob, os
import imageio
import torch
from torch.nn import functional as F
import numpy as np
import math

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_A', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--output_path', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--window_size', type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    list_gt, list_A = get_image_list(args)
    sum_mse = torch.zeros(args.window_size*args.window_size).to(device)
    count = 0
    for file_gt, file_A in zip(list_gt, list_A):
        flist = [file_gt, file_A]
        img_gt, img_A = get_cropped_images(window_size=args.window_size, device=device, files=flist)

        img_mse = calculate_psnr(img_gt, img_A, device=device).unsqueeze(1)
        img_mse_fold = F.unfold(img_mse, kernel_size=args.window_size, stride=args.window_size).permute(2,0,1).squeeze(1)
        count += img_mse_fold.shape[0]
        sum_mse += torch.sum(img_mse_fold, dim=0)

    sum_mse = sum_mse / count
    sum_mse = sum_mse.reshape(args.window_size, args.window_size)
    psnr_mse = 20 * torch.log10(1 / torch.sqrt(sum_mse))
    print(psnr_mse)

def get_cropped_images(window_size, device, files):
    def _get_cropped_images(x):
        img = imageio.imread(x)
        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # HCW-RGB to CHW-RGB
        else:
            img = np.stack((img, img, img), axis=0)
        img = torch.from_numpy(img).float().unsqueeze(0).to(device) / 255  # CHW-RGB to NCHW-RGB
        h_old, w_old = img.shape[-2:]
        h_new, w_new = (h_old//window_size)*window_size, (w_old//window_size)*window_size
        img = img[..., :h_new, :w_new]
        return img
    
    return [_get_cropped_images(a) for a in files]

def get_image_list(args):
    list_gt = sorted(glob.glob(os.path.join(args.folder_gt, '*')))
    list_A = sorted(glob.glob(os.path.join(args.folder_A, '*')))
    return list_gt, list_A

def calculate_psnr(img1, img2, device, y_psnr=True):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    if y_psnr and img1.shape[-1] > 1:
        gray_coeffs = torch.Tensor([65.738, 129.057, 25.064]) / 256
        gray_coeffs = gray_coeffs.reshape(1,-1,1,1).to(device)
        img1 = torch.sum(img1 * gray_coeffs, dim=1)
        img2 = torch.sum(img2 * gray_coeffs, dim=1)

    img_mse = (img1 - img2)**2
    return img_mse


if __name__=='__main__':
    main()
