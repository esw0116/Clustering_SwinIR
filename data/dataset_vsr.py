import random
import numpy as np
import glob, os
import imageio
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetVSR(data.Dataset):
    def __init__(self, opt):
        super(DatasetVSR, self).__init__()
        self.opt = opt
        self.train = True if self.opt['phase'] == 'train' else False
        self.scale = [opt['scale']] if opt['scale'] else [4]
        self.n_frames = opt['n_frames'] if opt['n_frames'] else 5
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.images_hr, self.images_lr = self._scan()
        
        if self.train:
            n_patches = opt['dataloader_batch_size'] * 1000
            n_images = len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.opt['dataroot_H'], '*', '*.png'))
        )
        names_lr = [[] for _ in self.scale]
        for si, s in enumerate(self.scale):
            if s == 1:
                names_lr[si]=names_hr

            else:
                for i, f in enumerate(names_hr):
                    filename, _ = os.path.splitext(os.path.basename(f))
                    foldername = os.path.basename(os.path.dirname(f))

                    f_list = sorted(glob.glob(os.path.join(self.opt['dataroot_L'], 'X{}'.format(s), foldername, '*')))
                    select_idx = util.index_generation(int(filename), len(f_list), self.n_frames,
                                            padding='new_info')
                    lr_list = [f_list[i] for i in select_idx]
                    names_lr[si].append(lr_list)
                    
        return names_hr, names_lr

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair = util.set_channel(*pair, n_channels=3)
        pair_t = util.np2Tensor(*pair, rgb_range=1)
        pair_lr = torch.cat(pair_t[:-1], dim=0)
        pair_hr = pair_t[-1]

        return {'L': pair_lr, 'H': pair_hr, 'L_path': filename, 'H_path': filename}

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file_hr(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)

        return hr, filename
    
    def _load_rain_test(self, idx):
        f_hr = self.derain_hr_test[idx]
        f_lr = self.derain_lr_test[idx]
        filename, _ = os.path.splitext(os.path.basename(f_lr))
        norain = imageio.imread(f_hr)
        rain = imageio.imread(f_lr)
        return norain, rain, filename
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
    
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[0][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        hr = imageio.imread(f_hr)
        lr = []
        for f in f_lr:
            lr.append(imageio.imread(f))
        return lr, hr, filename
    
    def _load_file_deblur(self, idx, train = True):
        idx = self._get_index(idx)
        if train:
            f_hr = self.images_hr[idx]
            f_lr = self.images_lr[idx]
        else:
            f_hr = self.deblur_hr_test[idx]
            f_lr = self.deblur_lr_test[idx]
        
        filename, _ = os.path.splitext(os.path.basename(f_hr))
        filename = f_hr[-27:-17] + filename
        hr = imageio.imread(f_hr)
        lr = imageio.imread(f_lr)

        return lr, hr, filename

    def get_patch_hr(self, hr):
        scale = self.scale[0]
        if self.train:
            hr = self.get_patch_img_hr(
                hr,
                patch_size=self.patch_size,
                scale=1
            )

        return hr

    def get_patch_img_hr(self, img, patch_size=96, scale=2):
        ih, iw = img.shape[:2]

        tp = patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        ret = img[iy:iy + ip, ix:ix + ip, :]

        return ret

    
    def get_patch(self, lr, hr):
        scale = self.scale[0]
        if self.train:
            imgs = util.get_patch(
                *lr, hr,
                patch_size=self.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1)
            )
            # if not self.args.no_augment: 
            imgs = util.augment(*imgs)
            
            return imgs

        else:
            ih, iw = lr[0].shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

            return [*lr, hr]
