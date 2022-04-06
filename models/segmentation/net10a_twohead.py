import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import io
import torchvision.transforms.functional as tf

__all__ = ["SegmentationNet10aTwoHead"]

class VGGTrunk(nn.Module):
  def __init__(self):
    super(VGGTrunk, self).__init__()

  def _make_layers(self, batch_norm=True):
    layers = []
    in_channels = self.in_channels
    for tup in self.cfg:
      assert (len(tup) == 2)

      out, dilation = tup
      sz = self.conv_size
      stride = 1
      pad = self.pad  # to avoid shrinking

      if out == 'M':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      elif out == 'A':
        layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
      else:
        conv2d = nn.Conv2d(in_channels, out, kernel_size=sz,
                           stride=stride, padding=pad,
                           dilation=dilation, bias=False)
        if batch_norm:
          layers += [conv2d, nn.BatchNorm2d(out,
                                            track_running_stats=self.batchnorm_track),
                     nn.ReLU(inplace=True)]
        else:
          layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = out

    return nn.Sequential(*layers)


class VGGNet(nn.Module):
  def __init__(self):
    super(VGGNet, self).__init__()

  def _initialize_weights(self, mode='fan_in'):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        assert (m.track_running_stats == self.batchnorm_track)
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class SegmentationNet10aTrunk(VGGTrunk):
  def __init__(self, cfg):
    super(SegmentationNet10aTrunk, self).__init__()

    # self.batchnorm_track = config.batchnorm_track
    self.batchnorm_track = True
    # assert (config.input_sz % 2 == 0)

    self.conv_size = 3
    self.pad = 1
    self.cfg = cfg
    #self.in_channels = config.in_channels if hasattr(config, 'in_channels') else 3
    self.in_channels = 4

    self.features = self._make_layers()

  def forward(self, x):
    x = self.features(x)  # do not flatten
    return x


class SegmentationNet10aHead(nn.Module):
  def __init__(self, output_k, cfg):
    super(SegmentationNet10aHead, self).__init__()

    # self.batchnorm_track = config.batchnorm_track
    self.batchnorm_track = True

    self.cfg = cfg
    num_features = self.cfg[-1][0]

    # self.num_sub_heads = config.num_sub_heads
    self.num_sub_heads = 1

    self.heads = nn.ModuleList([nn.Sequential(
      nn.Conv2d(num_features, output_k, kernel_size=1,
                stride=1, dilation=1, padding=1, bias=False),
      nn.Softmax2d()) for _ in range(self.num_sub_heads)])

    # self.input_sz = config.input_sz
    # self.input_sz = 256

  def forward(self, x, input_size):
    results = []
    for i in range(self.num_sub_heads):
      x_i = self.heads[i](x)
      x_i = F.interpolate(x_i, size=input_size, mode="bilinear")
      results.append(x_i)

    return results


class SegmentationNet10aTwoHead(VGGNet):
  cfg = [(64, 1), (128, 1), ('M', None), (256, 1), (256, 1),
         (512, 2), (512, 2)]  # 30x30 recep field
  def __init__(self):
    super(SegmentationNet10aTwoHead, self).__init__()

    # self.batchnorm_track = config.batchnorm_track
    self.batchnorm_track = True

    self.trunk = SegmentationNet10aTrunk(cfg=SegmentationNet10aTwoHead.cfg)
    self.head_A = SegmentationNet10aHead(output_k=36,
                                         cfg=SegmentationNet10aTwoHead.cfg)
    self.head_B = SegmentationNet10aHead(output_k=6,
                                         cfg=SegmentationNet10aTwoHead.cfg)

    self._initialize_weights()

  def forward(self, x, input_size, head="B"):
    x = self.trunk(x)
    if head == "A":
      x = self.head_A(x, input_size)
    elif head == "B":
      x = self.head_B(x, input_size)
    else:
      assert (False)

    return x


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_ind", type=int, required=True)
  parser.add_argument("--arch", type=str, required=True)
  parser.add_argument("--opt", type=str, default="Adam")
  parser.add_argument("--mode", type=str, default="IID")  # or IID+

  parser.add_argument("--dataset", type=str, required=True)
  parser.add_argument("--dataset_root", type=str, required=True)

  parser.add_argument("--use_coarse_labels", default=False,
                      action="store_true")  # COCO, Potsdam
  parser.add_argument("--fine_to_coarse_dict", type=str,  # COCO
                      default="/users/xuji/iid/iid_private/code/datasets"
                              "/segmentation/util/out/fine_to_coarse_dict.pickle")
  parser.add_argument("--include_things_labels", default=False,
                      action="store_true")  # COCO
  parser.add_argument("--incl_animal_things", default=False,
                      action="store_true")  # COCO
  parser.add_argument("--coco_164k_curated_version", type=int, default=-1)  # COCO

  parser.add_argument("--gt_k", type=int, required=True)
  parser.add_argument("--output_k_A", type=int, required=True)
  parser.add_argument("--output_k_B", type=int, required=True)

  parser.add_argument("--lamb_A", type=float, default=1.0)
  parser.add_argument("--lamb_B", type=float, default=1.0)

  parser.add_argument("--lr", type=float, default=0.01)
  parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
  parser.add_argument("--lr_mult", type=float, default=0.1)

  parser.add_argument("--use_uncollapsed_loss", default=False,
                      action="store_true")
  parser.add_argument("--mask_input", default=False, action="store_true")

  parser.add_argument("--num_epochs", type=int, default=1000)
  parser.add_argument("--batch_sz", type=int, required=True)  # num pairs
  parser.add_argument("--num_dataloaders", type=int, default=3)
  parser.add_argument("--num_sub_heads", type=int, default=5)

  parser.add_argument("--out_root", type=str,
                      default="/scratch/shared/slow/xuji/iid_private")
  parser.add_argument("--restart", default=False, action="store_true")

  parser.add_argument("--save_freq", type=int, default=5)
  parser.add_argument("--test_code", default=False, action="store_true")

  parser.add_argument("--head_B_first", default=False, action="store_true")
  parser.add_argument("--batchnorm_track", default=False, action="store_true")

  # data transforms
  parser.add_argument("--no_sobel", default=False, action="store_true")

  parser.add_argument("--include_rgb", default=False, action="store_true")
  parser.add_argument("--pre_scale_all", default=False,
                      action="store_true")  # new
  parser.add_argument("--pre_scale_factor", type=float, default=0.5)  #

  parser.add_argument("--input_sz", type=int, default=161)  # half of kazuto1011

  parser.add_argument("--use_random_scale", default=False,
                      action="store_true")  # new
  parser.add_argument("--scale_min", type=float, default=0.6)
  parser.add_argument("--scale_max", type=float, default=1.4)

  # transforms we learn invariance to
  parser.add_argument("--jitter_brightness", type=float, default=0.4)
  parser.add_argument("--jitter_contrast", type=float, default=0.4)
  parser.add_argument("--jitter_saturation", type=float, default=0.4)
  parser.add_argument("--jitter_hue", type=float, default=0.125)

  parser.add_argument("--flip_p", type=float, default=0.5)

  parser.add_argument("--use_random_affine", default=False,
                      action="store_true")  # new
  parser.add_argument("--aff_min_rot", type=float, default=-30.)  # degrees
  parser.add_argument("--aff_max_rot", type=float, default=30.)  # degrees
  parser.add_argument("--aff_min_shear", type=float, default=-10.)  # degrees
  parser.add_argument("--aff_max_shear", type=float, default=10.)  # degrees
  parser.add_argument("--aff_min_scale", type=float, default=0.8)
  parser.add_argument("--aff_max_scale", type=float, default=1.2)

  # local spatial invariance. Dense means done convolutionally. Sparse means done
  #  once in data augmentation phase. These are not mutually exclusive
  parser.add_argument("--half_T_side_dense", type=int, default=0)
  parser.add_argument("--half_T_side_sparse_min", type=int, default=0)
  parser.add_argument("--half_T_side_sparse_max", type=int, default=0)

  config = parser.parse_args()
  # net = SegmentationNet10aTwoHead(config)
  net = SegmentationNet10aTwoHead()
  state = torch.load('model_544.pytorch')
  net.load_state_dict(state)

  print(f'# Parameters: {sum(p.nelement() for p in net.parameters())}')
  breakpoint()
  net.cuda()

  x = io.read_image('baby.png')
  # x = x[:, :16, :16]
  grey = tf.to_tensor(tf.to_grayscale(tf.to_pil_image(x)))

  x = x.float()
  x /= 255
  
  x = torch.cat((x, grey), dim=0)
  x.unsqueeze_(0)
  x = x.cuda()

  # x = transforms.sobel_process(x, True)
  label = net(x)[0]
  print(label.shape)
  _, c = label.max(dim=1)
  c *= 42
  c = c.byte()
  c = c.cpu()
  print(c.shape)
  io.write_png(c, 'label.png')
  # imageio.imwrite('labels.png', c[0].numpy())