import typing

import torch
from torch import autograd
from torch.autograd import function
from torch.cuda import amp

import srwarp_cuda


def check_cuda(x: torch.Tensor, name: str) -> None:
    if not x.is_cuda:
        raise ValueError('{} must be on a GPU!'.format(name))

    return

def gather_2d(labels: torch.Tensor, k: int) -> torch.Tensor:
    check_cuda(labels, 'labels')
    b, l = labels.size()
    attn_labels = labels.new_zeros(b, l, l)
    srwarp_cuda.gather_2d(b, l, k, labels, attn_labels)
    return attn_labels