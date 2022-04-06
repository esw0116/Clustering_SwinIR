import torch.nn as nn
import torch.nn.functional as F

#from ..cluster.vgg import VGGTrunk, VGGNet
from vgg import VGGTrunk, VGGNet

__all__ = ["SegmentationNet10a"]


# From first iteration of code, based on VGG11:
# https://github.com/xu-ji/unsup/blob/master/mutual_information/networks
# /vggseg.py

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
      print(x.shape)
      x_i = self.heads[i](x)
      print(x_i.shape)
      x_i = F.interpolate(x_i, size=input_size, mode="bilinear")
      results.append(x_i)

    return results
