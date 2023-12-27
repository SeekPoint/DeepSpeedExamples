#!/usr/bin/env python3

import os
import argparse

import torch
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms
from torchvision.models import AlexNet
from torchvision.models import vgg19

import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.utils import RepeatingLoader

from pydebug import gd, infoTensor
# pid = os.getpid()
def cifar_trainset(local_rank, dl_path='/tmp/cifar10-data'):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    gd.debuginfo(prj="ds_chat", info=f'transform={transform}')

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()

    if local_rank != 0:
        dist.barrier()
    '''
    dist.barrier() pytorch在分布式训练过程中，对于数据的读取是采用主进程预读取并缓存，然后其它进程从缓存中读取，
    不同进程之间的数据同步具体通过torch.distributed.barrier()实现。

    在上面的代码示例中，如果执行 cifar_trainset() 函数的进程不是主进程，
    即rank不等于0，会执行相应的 torch.distributed.barrier()，
    设置一个阻塞栅栏，让此进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）；
    如果执行create_dataloader()函数的进程是主进程，其会直接去读取数据并处理，
    然后其处理结束之后会接着遇到torch.distributed.barrier()，
    此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。
    '''

    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)

    if local_rank == 0:
        dist.barrier()

    return trainset


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline_parallel_size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', type=int, default=1138, help='PRNG seed')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def train_base(args):
    torch.manual_seed(args.seed)

    # VGG also works :-)
    #net = vgg19(num_classes=10)
    net = AlexNet(num_classes=10)

    gd.debuginfo(prj="ds_chat", info=f'alex net={net}')

    trainset = cifar_trainset(args.local_rank)

    gd.debuginfo(prj="ds_chat", info=f'trainset={trainset}')

    tmp = []
    for p in net.parameters():
        if p.requires_grad:
            tmp.append(p)
            gd.debuginfo(prj="ds_chat", info=f'base net req_grad p={infoTensor(p)}')
        else:
            gd.debuginfo(prj="ds_chat", info=f'base net Not req_grad p={infoTensor(p)}')

    gd.debuginfo(prj="ds_chat", info=f'len of tmp={len(tmp)}')

    engine, _, dataloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=tmp,
        training_data=trainset)

    gd.debuginfo(prj="ds_chat", info=f'engine={engine}')
    gd.debuginfo(prj="ds_chat", info=f'1-dataloader={dataloader}')
    
    # from calltrace import g_ct # 不要放在函数外！！注意导入时已经开始startrecord

    dataloader = RepeatingLoader(dataloader)
    gd.debuginfo(prj="ds_chat", info=f'2-dataloader={dataloader}')

    data_iter = iter(dataloader)

    gd.debuginfo(prj="ds_chat", info=f'data_iter={data_iter}')

    rank = dist.get_rank()
    gd.debuginfo(prj="ds_chat", info=f'rank={rank}')

    gas = engine.gradient_accumulation_steps()
    gd.debuginfo(prj="ds_chat", info=f'gas={gas}')

    criterion = torch.nn.CrossEntropyLoss()
    gd.debuginfo(prj="ds_chat", info=f'criterion={criterion}')

    total_steps = args.steps * engine.gradient_accumulation_steps()

    gd.debuginfo(prj="ds_chat", info=f'total_steps={total_steps}')

    step = 0
    for micro_step in range(total_steps):

        batch = next(data_iter)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)

        gd.debuginfo(prj="ds_chat", info=f'batch={batch}')
        gd.debuginfo(prj="ds_chat", info=f'inputs={inputs}')
        gd.debuginfo(prj="ds_chat", info=f'labels={labels}')

        outputs = engine(inputs)
        gd.debuginfo(prj="ds_chat", info=f'outputs={outputs}')

        loss = criterion(outputs, labels)
        gd.debuginfo(prj="ds_chat", info=f'loss={loss}')

        engine.backward(loss)
        gd.debuginfo(prj="ds_chat", info=f'--------------------------------------------')
        engine.step()

        if micro_step % engine.gradient_accumulation_steps() == 0:
            step += 1
            if rank == 0 and (step % 10 == 0):
                print(f'step: {step:3d} / {args.steps:3d} loss: {loss}')


# 将 视觉上的直观 layers 保存为 feature, avgpool,classifier 为顺序的数组传入 pipeModule 中
def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]

    gd.debuginfo(prj="ds_chat", info=f'layers={layers}')

    return layers


def train_pipe(args, part='parameters'):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)

    #
    # Build the model
    #

    # VGG also works :-)
    #net = vgg19(num_classes=10)

    logf = f'pipeline_build_AlexNet'
    #if args.local_rank == 0:
    # gd.enable(info=logf)
    gd.emb_start(info=logf)

    ori_net = AlexNet(num_classes=10)
    gd.debuginfo(prj="ds_chat", info=f'ori_net={ori_net}')

    #if args.local_rank == 0:
    # gd.disable(info=logf)
    gd.emb_end(info=logf)

    logf = f'pipeline_PipelineModule'
    #if args.local_rank == 0:
    # gd.enable(info=logf)
    gd.emb_start(info=logf)

    ppnet = PipelineModule(layers=join_layers(ori_net),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         partition_method=part,
                         activation_checkpoint_interval=0)
    gd.debuginfo(prj="ds_chat", info=f'ppnet={ppnet}')

    #if args.local_rank == 0:
    # gd.disable(info=logf)
    gd.emb_end(info=logf)

    logf = f'pipeline_trainset_args'
    # if args.local_rank == 0:
    # gd.enable(info=logf)
    gd.emb_start(info=logf)

    trainset = cifar_trainset(args.local_rank)

    gd.debuginfo(prj="ds_chat", info=f'trainset={trainset}')
    gd.debuginfo(prj="ds_chat", info=f'B-args={args}')

    tmp = []
    for p in ppnet.parameters():
        if p.requires_grad:
            tmp.append(p)
            gd.debuginfo(prj="ds_chat", info=f'PP NET req_grad p={infoTensor(p)}')
        else:
            gd.debuginfo(prj="ds_chat", info=f'PP NET Not req_grad p={infoTensor(p)}')

    gd.debuginfo(prj="ds_chat", info=f'len of model_p={len(tmp)}')

    # if args.local_rank == 0:
    # gd.disable(info=logf)
    gd.emb_end(info=logf)

    # 内部分开记录log
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=ppnet,
        model_parameters= tmp,
        training_data=trainset)

    gd.debuginfo(prj="ds_chat", info=f'engine={engine}')

    for step in range(args.steps):
        logf = f'pipeline_step{step:04}'

        #if args.local_rank == 0:
        # gd.enable(info=logf)
        gd.emb_start(info=logf)

        loss = engine.train_batch()
        gd.debuginfo(prj="ds_chat", info=f'step={step}, loss={loss}')

        #if args.local_rank == 0:
        # gd.disable(info=logf)
        gd.emb_end(info=logf)

if __name__ == '__main__':
    args = get_args()

    logf = f'pipeline_deepspeed.init_distributed'
    # if args.local_rank == 0:
    #     gd.enable(info=logf)
    gd.emb_start(info=logf)

    gd.debuginfo(prj="ds_chat", info=f'A-args={args}')
    deepspeed.init_distributed(dist_backend=args.backend)

    # if args.local_rank == 0:
    #     gd.disable(info=logf)
    gd.emb_end(info=logf)

    gd.debuginfo(prj="ds_chat", info=f'--------------------------------------------')

    # 通过 os 函数 os.getenv('RANK') 可以查看当前函数调用所在的卡
    args.local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.local_rank)

    if args.pipeline_parallel_size == 0:
        gd.debuginfo(prj="ds_chat")
        train_base(args)
    else:
        gd.debuginfo(prj="ds_chat")
        train_pipe(args)


