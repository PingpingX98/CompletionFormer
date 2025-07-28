"""
    CompletionFormer
    ======================================================================

    main script for training and testing.
"""


from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = args_config.address
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
torch.autograd.set_detect_anomaly(True)

import utility
from model.completionformer import CompletionFormer
from summary.cfsummary import CompletionFormerSummary
from summary.cfsummarynew import CompletionFormerSummarynew
from metric.cfmetric import CompletionFormerMetric
from metric.cfmetricnew import CompletionFormerMetricnew
from data import get as get_data
from loss.l1l2loss import L1L2Loss

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import torch.utils.benchmark as benchmark
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Minimize randomness
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args



def test(args):
    # Prepare dataset
    data = get_data(args)

    data_test = data(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    # Network
    if args.model == 'CompletionFormer':
        net = CompletionFormer(args)
    else:
        raise TypeError(args.model, ['CompletionFormer',])
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError
        print('Checkpoint loaded from {}!'.format(args.pretrain))

    net = nn.DataParallel(net)

    #metric = CompletionFormerMetric(args)
    metric_new = CompletionFormerMetricnew(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
    except OSError:
        pass

    #writer_test = CompletionFormerSummary(args.save_dir, 'test', args, None, metric.metric_name)
    writer_test_new = CompletionFormerSummarynew(args.save_dir, 'test', args, None, metric_new.metric_name)

    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)
    json_dir = '/data/result'
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, 'inference_time.json')
    t_total = 0
    times = []
    init_seed()
    for batch, sample in enumerate(loader_test):
        # sample = {key: val.cuda() for key, val in sample.items()
        #           if val is not None}
        sample = {
        key: [v.cuda() for v in val] if isinstance(val, list) else val.cuda()
        for key, val in sample.items()
        if val is not None
    }
        timer = benchmark.Timer(
            stmt='net(sample)',
            globals={'net': net, 'sample': sample}
        )
        measured_time = timer.timeit(1000).mean
        print(f"{measured_time} sec")
       
        break
    
        t0 = time.time()
        with torch.no_grad():
            output = net(sample)
        t1 = time.time()

        t_total += (t1 - t0)

        #metric_val = metric.evaluate(sample, output, 'test')
        metric_val_new = metric_new.evaluate(sample, output, 'test')

        #writer_test.add(None, metric_val)
        writer_test_new.add(None, metric_val_new)

        # Save data for analysis
        if args.save_result_only:
            #writer_test.save(args.epochs, batch, sample, output)
            writer_test_new.save(args.epochs, batch, sample, output)

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        # error_str = '{} | Test'.format(current_time)
        error_str = '{} | Test | Timer average: {:.6f} s'.format(current_time, measured_time)
        if batch % args.print_freq == 0:
            pbar.set_description(error_str)
            pbar.update(loader_test.batch_size)

    pbar.close()

 

def main(args):
    init_seed()
    if not args.test_only:
        if args.no_multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

        args.pretrain = '{}/model_{:05d}.pt'.format(args.save_dir, args.epochs)

    test(args)


if __name__ == '__main__':
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')

    main(args_main)