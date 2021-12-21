import os
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='run multiple gpu setting')
    parser.add_argument('--conf', default='',
                        type=str)
    args = parser.parse_args()
    conf = args.conf

    n = torch.cuda.device_count()
    print(n)

    command = 'python -m torch.distributed.launch --nproc_per_node=%d --use_env main_train_psnr.py %s' % (n, conf)
    print (command)
    os.system(command)

if __name__ == '__main__':
    main()