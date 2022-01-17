import torch
from torch import cuda

from srwarp import svf
import numpy as np
import tqdm

def main():
    torch.set_printoptions(edgeitems=5, linewidth=160)
    test_input = torch.load('debug_input.pth')
    labels = test_input['labels'].contiguous()
    b = test_input['b']
    l = test_input['l']
    k = 16

    print(f'labels: {labels.size()} / b: {b} / l: {l}')
    for idx in range(100):
        t1 = cuda.Event(enable_timing=True)
        t1.record()
        cuda.synchronize()

        labels_f = labels.flatten(0,1)
        grid_x, grid_y = torch.meshgrid(labels_f*k, labels_f)
        grid_xy = grid_x + grid_y
        s=(range(b), np.s_[:], range(b), np.s_[:])
        attn_labels = grid_xy.reshape(b, l, b, l)[s]

        t2 = cuda.Event(enable_timing=True)
        t2.record()
        cuda.synchronize()

        #attn_labels_2 = torch.zeros((b, l, l)).type_as(labels)
        attn_labels_2 = labels.new_zeros(b, l, l)
        for i in range(b):
            grid_x, grid_y = torch.meshgrid(labels[i]*16, labels[i])
            attn_labels_2[i, :, :] = grid_x + grid_y

        t3 = cuda.Event(enable_timing=True)
        t3.record()
        cuda.synchronize()

        #labels_int = labels.int()
        attn_labels_3 = svf.gather_2d(labels, k)
        #attn_labels_3 = attn_labels_3.long()

        t4 = cuda.Event(enable_timing=True)
        t4.record()
        cuda.synchronize()

        '''
        attn_labels_4 = labels.new_zeros(b, l, l)
        for ii in tqdm.trange(64, ncols=80):
            for jj in range(256):
                for kk in range(256):
                    class_ij = labels[ii][jj]
                    class_ik = labels[ii][kk]
                    attn_labels_4[ii][jj][kk] = k * class_ij + class_ik
        print('Reference labels')
        print(attn_labels)

        print('Cuda labels')
        print(attn_labels_3)

        #print('Hand labels')
        #print(attn_labels_4)
        '''

        #print(labels.size())
        if idx == 99:
            print(f'Index: {idx}')
            print(f'Faster: {t1.elapsed_time(t2)}')
            print(f'Old: {t2.elapsed_time(t3)}')
            print(f'Cuda: {t3.elapsed_time(t4)}')
            print(f'Error: {(attn_labels - attn_labels_2).pow(2).sum()}')
            print(f'Error cuda: {(attn_labels - attn_labels_3).pow(2).sum()}')
            print()

    return

if __name__ == '__main__':
    main()