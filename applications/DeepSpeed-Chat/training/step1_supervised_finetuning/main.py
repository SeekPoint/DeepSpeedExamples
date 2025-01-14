#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# RandomSampler和SequentialSampler（用于数据抽样）以及DistributedSampler（用于在分布式设置中进行数据抽样）。
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# pid = os.getpid()

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import mem_estimate_log

from pydebug import gd, infoTensor

# 因果语言建模的模型（AutoModelForCausalLM），优化器调度类型（SchedulerType），
# 默认的数据整理函数（default_data_collator）和获取优化器调度器的函数（get_scheduler）。
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler)

# logf = f'ph1-import_deepspeed'
# gd.enable(info=logf) --没有记录
import deepspeed
# gd.disable(info=logf)

# logf = f"ph1-import_deepspeed.ops.adam"
# gd.enable(info=logf) --没有记录
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
# gd.disable(info=logf)

# logf = f"ph1-import_deepspeed.comm"
# gd.enable(info=logf)   --没有记录
import deepspeed.comm as dist
# gd.disable(info=logf)

# 将当前脚本的父目录添加到系统路径中，以便可以从该目录下的utils目录导入一些自定义函数和模块。
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# 从utils目录中导入了一些自定义模块和函数，
# 包括数据处理函数（create_prompt_dataset），打印和设备转换函数（print_rank_0和to_device），
# 模型保存函数（save_hf_format），随机种子设置函数（set_random_seed），求平均函数（get_all_reduce_mean），
# 获取优化器参数组的函数（get_optimizer_grouped_parameters），
# 保存和加载模型的函数（save_zero_three_model和load_hf_tokenizer），
# 以及创建模型和处理模型的函数（create_hf_model）。



gd.debuginfo(prj="ds_chat", info=f"1==={os.path.pardir}")
gd.debuginfo(prj="ds_chat", info=f"2==={os.path.dirname(__file__)}")
gd.debuginfo(prj="ds_chat", info=f"3==={os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))}")
#
# 1=== ..
# 2=== /home/amd00/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
# 3=== /home/amd00/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/training


from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer, debugOGP
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model


def parse_args():
    # 创建一个argparse的解析器对象，这个对象可以添加命令行参数和处理它们。
    # description参数提供了一个对程序的简单描述。
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    # 1 训练数据集的路径
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')

    # 2 数据集的切分比例
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                             'phase 1, 2, and 3 data. For example the split `6,2,2`'
                             'will use 60% of data for phase 1, 20% for phase 2'
                             'and 20% for phase 3.')

    # 3 只用于SFT阶段的数据集路径  yknote???
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')

    # 4 数据相关文件的存储路径
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='./data_files/',
        help=
        'Where to store the data-related files such as shuffle index. \
        This needs to be on a local storage of a node (not on a shared storage)'
    )

    # 5 预训练模型的路径 或者来自 huggingface.co/models的 model identifier
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    # 6 每个设备上的训练批次大小
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    # 7 每个设备上的评估批次大小
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    # 8 最大序列长度
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    # 9 warmup之后的初始学习率
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
	
    # 10 权重衰减率
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")

    # 11 训练的次数epochs
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")

    # 12 在执行反向传播之前累积的更新步骤数
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )

    # 学习率调度器类型
    # 作用：根据预设的策略来动态调整学习率
    # ① linear: 线性调度器在每个训练步骤中将学习率按线性规则降低。
    # ② cosine: 余弦调度器根据余弦退火调度来调整学习率。
    #    学习率将在周期内从初始学习率线性减少到最小学习率，然后在下一个周期内重复这个过程。
    # ③ cosine_with_restarts: 这是余弦退火调度的一个变种，
    #    它在每个周期结束后重启学习率，使得每个周期的开始学习率总是高于结束时的学习率。
    # ④ polynomial: 多项式调度器将根据一个多项式规则来降低学习率，通常是一个衰减因子乘以训练步数的某个幂。
    # ⑤ constant: 常数调度器使学习率保持不变，这意味着在整个训练过程中，学习率不会被调整。
    # ⑥ constant_with_warmup: 这是常数调度器的一个变种，在一开始的几个步骤（即预热阶段）内，
    #    学习率会被线性增加到预设的学习率，然后在剩余的训练步骤中保持不变。
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )

    # 14 学习率调度器中的warmup步骤数
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")

    # 15 存储模型的位置
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")

    # 16 随机数seed，固定它可以重现模型
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")

    # 17 用于在GPU上进行分布式训练的local rank
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # 18 是否启用梯度checkpoint， 
    '''
    action=‘store_true’/‘store_false’。
    顾名思义，store_true就代表着一旦有这个参数，做出动作“将其值标为True”，也就是没有时，默认状态下其值为False
    '''
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')

    # 19 是否禁用模型的dropout
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')

    # deepspeed features
    # 20 是否启用ZeRO Offload技术
    # 是否使用ZeRO Offload技术。如果为True，那么将模型参数和优化器状态offload到CPU，否则不使用offload。
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')

    # 21 用于Actor模型（和clones）的ZeRO优化阶段
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')  #yknote---clone？？？

    ## LoRA for efficient training setting
    # 22 如果大于0，使用LoRA进行高效训练
    parser.add_argument("--lora_dim",
                        type=int,
                        default=1,
                        help="If > 0, use LoRA for efficient training.")

    # 23 LoRA的范围
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")

    # 24 只优化LoRA参数
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')

    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    # Tensorboard 路径
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')

    print('parser--1:', parser)
    # 这一行将DeepSpeed的配置参数添加到解析器中。

    # if dist.get_rank() == 0:  ==dist没有初始化，不能使用
    # logf = f'ph1-deepspeed.add_config_arguments'
    # gd.enable(info=logf)  --没有记录
    parser = deepspeed.add_config_arguments(parser)
    # gd.disable(info=logf)

    print('parser--2:', parser)  #yknote-TBD,比较变化

    # 这一行解析命令行参数并将它们存储在args对象中
    args = parser.parse_args()

    # Validate settings
    # 在这个代码块中，验证一些特定的参数设置是否合法。
    # 例如，如果同时启用了gradient_checkpointing和仅优化LoRA参数，那么将会抛出一个错误。
    # 如果启用了gradient_checkpointing并且lora_dim大于0，那么必须禁用只优化LoRA参数。
    # 这是因为梯度检查点和只优化LoRA参数这两个选项在同时启用时可能会引发冲突。
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args

# 这个函数是主函数，是训练语言模型的主流程，主要步骤包括解析命令行参数、
# 设置设备、准备数据、定义模型、配置优化器和学习率调度器、进行训练和评估等。
def main():
    # 第1步：超参数配 解析命令行参数。
    args = parse_args()
    gd.debuginfo(prj="ds_chat", info=f"args is={args}")  #yknote---TBD

    # 如果local_rank为-1，说明不在分布式训练环境下，设备设置为"cuda"；
    # 否则根据args.local_rank设置对应的cuda设备，并初始化分布式训练。
    if args.local_rank == -1:
        # 单机版的CUDA
        device = torch.device("cuda")
    else:
        # 分布式训练
        # 设置了当前进程中的默认device，确保每个进程在正确的device上运行
        torch.cuda.set_device(args.local_rank)

        # 确保tensor被创建或移动到正确的device上
        device = torch.device("cuda", args.local_rank)
        gd.debuginfo(prj="ds_chat", info=f"device is--1={device}") #yknote---TBD.

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        # 初始化分布式训练环境

        #if dist.get_rank() == 0: # 还没初始化，不能使用dist！！
        #if args.local_rank == 0:
        logf = f'ph1_z{args.zero_stage}_deepspeed.init_distributed'
        # gd.enable(info=logf)
        gd.emb_start(info=logf)
        #日志分析，主要就是初始化后端，目前用的是torch的

        deepspeed.init_distributed()
        #if dist.get_rank() == 0: # dist初始化好了
        # gd.disable(info=logf)
        gd.emb_end(info=logf)

    # 获取当前运行设备在分布式训练环境中的全局rank
    args.global_rank = torch.distributed.get_rank()
    gd.debuginfo(prj="ds_chat", info=f"args.global_rank is={args.global_rank}")
    '''
    两张卡
    args.global_rank is: 1
    args.global_rank is: 0
    '''

    # 获取deepspeed的训练配置  根据输入参数返回一个训练数据集的配置字典
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    gd.debuginfo(prj="ds_chat", info=f"ph1 ds_config train is--1={ds_config}")

    # micro_batch训练是一种分布式训练技术，可以将一个大批次的数据分解成多个小批次，以适应GPU的内存限制
    # 在配置中设置训练时每个GPU的微批次大小和总的批次大小。
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size

    # 每个训练步骤中处理的数据总量
    # gradient_accumulation(梯度积累)是另一种应对内存限制的技术，它会在多个步骤中积累梯度，然后再一次性更新模型参数。
    # torch.distributed.get_world_size() ：在分布式训练环境中的节点（设备）数量
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    gd.debuginfo(prj="ds_chat", info=f"ph1 ds_config train is--2={ds_config}")
    gd.debuginfo(prj="ds_chat", info=f"ds_config['train_batch_size']is={ds_config['train_batch_size']}")  #8
    gd.debuginfo(prj="ds_chat", info=f"args.per_device_train_batch_size is={args.per_device_train_batch_size}") #4


    # If passed along, set the training seed now.
    # 设置随机种子以保证结果的可复现性。
    set_random_seed(args.seed)

    # 在此处添加一个barrier操作，确保所有进程都执行到这一点后再继续执行后续操作。
    # PyTorch分布式训练中的一个同步工具，它确保所有分布式进程都达到了这个阻塞点，
    # 然后再继续执行后面的代码，以避免出现某些进程快于其他进程的情况。
    torch.distributed.barrier()  #多机多卡很可能卡在这里！！！--比如版本pytorch不一致

    # 加载预训练模型tokenizer，fast_tokenizer=True表示使用优化过的、速度更快的tokenizer。
    # 加载预训练模型对应的分词器。
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    gd.debuginfo(prj="ds_chat", info=f"ph1 tokenizer --0={tokenizer}")

    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
	# 将tokenizer的填充方向设置为'right'，表示在序列的右侧（末尾）添加填充符号。
    tokenizer.padding_side = 'right'

    gd.debuginfo(prj="ds_chat", info=f"ph1 tokenizer --1={tokenizer}")

    # 创建预训练模型。 # 第2步：创建actor模型
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=args.disable_dropout)
    if args.zero_stage == 2 or args.zero_stage == 3:
        mem_estimate_log(args=args, exstr='-ph1-0', model = model, num_gpus_per_node=2, num_nodes=1)
    gd.debuginfo(prj="ds_chat", info=f"s1 model create_hf_model={model}")

    # 判断是否启用LoRA模式
    # 如果参数lora_dim大于0，将模型的线性层转换为LoRa层；如果只优化LoRa参数，关闭其他参数的梯度。
    if args.lora_dim > 0:
        '''
        yk--此处代码不同
        如果启用，则对名称中含有“decoder.layers.”且为线性层的结构部分引入LoRA旁路（实现先降维后升维的2个线性层），
        这类结构基本都是attention、信息交互用的inner线性层，
        这类结构的Weight参数将被冻结，转而优化LoRA旁路的参数。
        '''
        # 将模型中指定的线性层转换为LoRA层
        # lora_module_name指定了要转换的模块的名称 , lora_dim指定了LoRA的维度
        # if dist.get_rank() == 0:
        logf = f'ph1_z{args.zero_stage}_convert_linear_layer_to_lora'
        # gd.enable(info=logf)
        gd.emb_start(info=logf)

        model = convert_linear_layer_to_lora(model,
                                             args.lora_module_name,
                                             args.lora_dim)

        gd.debuginfo(prj="ds_chat", info = f"s1 convert_linear_layer_to_lora={model}")

        # if dist.get_rank() == 0:
        # gd.disable(info=logf)
        gd.emb_end(info=logf)

        if args.zero_stage == 2 or args.zero_stage == 3:
            mem_estimate_log(args=args, exstr='-ph1-1', model = model, num_gpus_per_node=2, num_nodes=1)

        if args.only_optimize_lora:
            # 将模型中非LoRA层的参数的requires_grad属性设为False，训练过程中只有LoRA层的参数会被更新/优化
            model = only_optimize_lora_parameters(model)
            gd.debuginfo(prj="ds_chat", info=f"s1 only_optimize_lora_parameters={model}")

            if args.zero_stage == 2 or args.zero_stage == 3:
                mem_estimate_log(args=args, exstr='-ph1-2', model = model, num_gpus_per_node=2, num_nodes=1)

    # 第3步：准备数据集（训练集和验证集）Prepare the data
    # 创建数据集和数据加载器：包括训练集和验证集，以及对应的采样器和数据加载器。
    train_phase = 1

    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank,
        args.data_path,
        args.data_split,
        args.data_output_path,
        train_phase,
        args.seed,
        tokenizer,
        args.max_seq_len,
        sft_only_data_path=args.sft_only_data_path)

    gd.debuginfo(prj="ds_chat", info=f"train_dataset={train_dataset}")
    gd.debuginfo(prj="ds_chat", info=f"eval_dataset={eval_dataset}")

    # DataLoaders creation
    if args.local_rank == -1:
        # 非分布式训练环境下，因此我们将使用随机采样和顺序采样
        # 在训练过程中，将随机选择样本进行训练，防止模型过拟合。
        train_sampler = RandomSampler(train_dataset)

        # 在评估过程中，将按照数据集中的顺序选择样本进行评估，验证集的顺序通常对模型的性能评估没有影响。
        eval_sampler = SequentialSampler(eval_dataset)

    else:
        # 在分布式训练环境中，将使用分布式采样
        # 创建一个用于训练集的分布式采样器，会在所有的训练节点上对样本进行均匀的分布，
        # 确保每个节点处理的样本是独立且均匀的，从而提高分布式训练的效率和稳定性。
        train_sampler = DistributedSampler(train_dataset)

        # 创建一个用于评估集的分布式采样器
        eval_sampler = DistributedSampler(eval_dataset)

    gd.debuginfo(prj="ds_chat", info=f"train_sampler={train_sampler}")
    gd.debuginfo(prj="ds_chat", info=f"eval_sampler={eval_sampler}")

    #default_data_collator 作用是将一批数据进行整合，使得它们可以整齐地堆叠在一起。

    # 创建用于训练集的数据加载器
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)

    # 创建用于评估集的数据加载器
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    gd.debuginfo(prj="ds_chat", info=f"train_dataloader={train_dataloader}")
    gd.debuginfo(prj="ds_chat", info=f"eval_dataloader={eval_dataloader}")

    # Split weights in two groups, one with weight decay and the other not.
	# 第4步：将模型参数分组、创建优化器 和 学习率调度器
	# 1. 将模型的参数分为两组，一组应用权重衰减，另一组不应用
    # 权重衰减是防止模型过拟合的一种策略，通常只对模型的权重参数应用。
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay)
	
    # gd.debuginfo(prj="ds_chat", info=f"optimizer_grouped_parameters={optimizer_grouped_parameters}")
    # 这里结构比较复杂，各种场景都有
    # for p in optimizer_grouped_parameters:
    #     for k, v in p.items():
    #         gd.debuginfo(prj="ds_chat", info=f"{k} is={infoTensor(v)}")
    # assert 0
    # debugOGP(optimizer_grouped_parameters) #因为在cpu上，即使使用print_rank0也打印了两次！
    gd.debuginfo(prj="ds_chat", info=f"len of optimizer_grouped_parameters={len(optimizer_grouped_parameters)}")

    # 根据是否使用DeepSpeed的CPU offload功能来选择优化器，优化器定义了如何更新模型的参数以最小化损失函数。

    # DeepSpeedCPUAdam : 配合DeepSpeed的CPU offload功能设计的，
    #          CPU offload可以将模型参数、优化器状态和梯度数据在CPU和GPU之间进行切换，以减轻GPU的内存压力。

    # FusedAdam : 它将一些计算操作融合在一起（fused），以减少计算时间和内存使用量。
    #          FusedAdam主要是为了提高在GPU上的运算效率。
	
    # 选择优化器类型，如果启用了梯度Offload，使用DeepSpeedCPUAdam，否则使用FusedAdam。
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
	
    gd.debuginfo(prj="ds_chat", info=f"AdamOptimizer={AdamOptimizer}")
    # AdamOptimizer : <class 'deepspeed.ops.adam.fused_adam.FusedAdam'>

    # 创建优化器
    # if dist.get_rank() == 0:
    logf = f'ph1_z{args.zero_stage}_AdamOptimizer_init'
    # gd.enable(info=logf)
    gd.emb_start(info=logf)

    # 日志分析，这里会触发c++的编译op过程！！是否重新编译是nvcc编译器决定的，和ds无关！
    # 触发 FusedAdamBuilder 构建 ->jit_load
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    gd.debuginfo(prj="ds_chat", info=f"optimizer={optimizer}")
    # if dist.get_rank() == 0:
    # gd.disable(info=logf)
    gd.emb_end(info=logf)

    # 计算每个epoch需要进行的更新步数，等于训练数据集大小除以梯度累积步数（对结果向上取整）
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    gd.debuginfo(prj="ds_chat", info=f"num_update_steps_per_epoch={num_update_steps_per_epoch}")

    # 创建学习率调度器。
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type, # 调度器的类型
        optimizer=optimizer, # 优化器，在每个训练步骤调整其学习率
        num_warmup_steps=args.num_warmup_steps, # 预热步骤数，在训练开始的一段时间内，
                                                # 学习率从0线性增加到预设的初始学习率，预热过程有助于模型的稳定训练。
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch, # 总的训练步骤数
    )

    gd.debuginfo(prj="ds_chat", info=f"lr_scheduler={lr_scheduler}")
    # lr_scheduler : <torch.optim.lr_scheduler.LambdaLR object at 0x7f469aea9fd0>

    # 第5步：deepspeed初始化，创建模型、优化器、学习率调度器

    # logf 记录拆分更细 目的在于更加有效的diff对比
    #if args.local_rank == 0:
    logf = f'ph1_z{args.zero_stage}_deepspeed.initialize_input'
    # gd.enable(info=logf)
    gd.emb_start(info=logf)

    gd.debuginfo(prj="ds_chat", info=f"model in={model}")
    gd.debuginfo(prj="ds_chat", info=f"optimizer in={optimizer}")
    # gd.debuginfo(prj="ds_chat", info=f"lr_scheduler in={lr_scheduler}")
    # lr_scheduler in=<torch.optim.lr_scheduler.LambdaLR object at 0x7ff60407d580>
    #if args.local_rank == 0:
    # gd.disable(info=logf)
    gd.emb_end(info=logf)

    # #触发 deepspeed/__init__.py f# initialize， 在内部进行 分段记录logf
    model, optimizer, tmpdumb, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config, # DeepSpeed的配置信息
        lr_scheduler=lr_scheduler, # 学习率调度器
        dist_init_required=True)  # 需要进行分布式训练的初始化

    #if args.local_rank == 0:
    logf = f'ph1_z{args.zero_stage}_deepspeed.initialize_output'
    # gd.enable(info=logf)
    gd.emb_start(info=logf)

    gd.debuginfo(prj="ds_chat", info=f"tmpdumb out deepspeed.initialize ={tmpdumb}")
    gd.debuginfo(prj="ds_chat", info=f"model out={model}")
    gd.debuginfo(prj="ds_chat", info=f"optimizer out={optimizer}")
    # gd.debuginfo(prj="ds_chat", info=f"lr_scheduler out={lr_scheduler}")
    # I# lr_scheduler out=<torch.optim.lr_scheduler.LambdaLR object at 0x7ff60407d580>  和上面in的完全一样！没有改变

    #if args.local_rank == 0:
    # gd.disable(info=logf)
    gd.emb_end(info=logf)

    # 如果启用了梯度检查点，那么在模型中也启用梯度检查点。
    if args.gradient_checkpointing:
        # 在模型中启用梯度检查点
        # # 梯度检查点是一种可以减少训练过程中内存使用的技术，通过牺牲一部分计算效率来换取内存占用的降低，会增加一些计算时间。
        model.gradient_checkpointing_enable()

    '''
    1.3.3 phase1的指标评估
    DeepSpeed-Chat选择了困惑度perplexity作为phase1训练期间的评估指标。
    需要注意的是，perplexity不是绝对的评估准则，
    甚至有可能perplexity评估结果与实际情况并不一致（即，perplexity已经处于较低水平，但模型的实际生成能力却仍然堪忧），
    这点DeepSpeed-Chat团队也有做出过说明。
    
    Supervised fine-tuning (SFT) 
    has indeed made significant progress in the field of large language models (LLMs). 
    However, unexpected behaviors such as repeating content generation and inconsistency 
    between perplexity (PPL) scores and generation capabilities can still occur.
    
    但无论如何，源码中phase1定义的evaluation是基于perplexity来进行的，我们仍有必要具体了解其实现过程。
    
    困惑度perplexity是一种度量语言模型性能的指标，它衡量了训练好的模型对测试数据的拟合程度，
    对于输出句子的每个token，都可以得到其输出的置信概率值，将这些值相乘并取其几何平均数的倒数即可计算得到困惑度perplexity，
    使用公式表达更为简洁：
        005.png
    其中，输出的句子共有T个token，第t个token的置信概率值为p_t
    
    而CausalLM模型的训练过程通常采用对数似然损失来进行优化，其输出的损失公式如下：
        006.png
    其中，输出的句子共有T个token，第t个token的置信概率值为p_t
    
    因此perplexity与CausalLM的loss之间实际存在如下关系：
        perplexity=exp(loss)
    
    相关源码的perplexity计算也是基于上述公式得到的：
    先是将验证数据输入至模型，得到模型loss输出，然后通过perplexity与loss之间的指数关系计算得到perplexity。

    '''
    ## 第6步：模型验证 模型评估 定义模型评估函数，用于计算模型在验证集上的困惑度。
    def evaluation(model, eval_dataloader):
        # 以困惑度perplexity为评估指标进行验证
        # 将模型切换为评估模式。
        model.eval() 
		
        # 初始化loss。
        losses = 0  
		
        # 对于评估数据集中的每一个batch。
        for step, batch in enumerate(eval_dataloader):  
            """
            batch: 由input_ids、attention_mask、labels共3个部分组成的dict。
                   其中每个部分的shape均为(bs, max_seq_len)
            """
            # 将batch数据移到对应的设备上。
            batch = to_device(batch, device)
			
            # 在此上下文管理器中，不计算梯度，这样可以节省存储和计算资源。
            with torch.no_grad():  
                # 将batch数据输入模型，进行前向计算。
				# 用当前批次的输入数据去前向传播模型，并得到模型的输出。
                outputs = model(**batch)

            """Causal LM 的损失函数为交叉熵损失"""
            # 取出模型的输出中的loss。
            loss = outputs.loss 
			
            # 将当前的loss累加到总的losses中， 这里losses是整个评估过程中的损失值
            losses += loss.float()  
        
        # 计算平均的loss。
        losses = losses / (step + 1)

        # 尝试计算模型的困惑度，如果捕捉到溢出错误，将困惑度设置为无穷大。
        try:
            # 计算模型的困惑度，这是评估语言模型性能的常用指标。困惑度的计算方法是对平均损失值取指数。
            # 如果这步运算没有发生溢出，那么困惑度的值就是torch.exp(losses)。
            # 当损失值过大时，指数运算可能会导致溢出
            perplexity = torch.exp(losses)
        except OverflowError:
            # 将困惑度的值设为无穷大。
            perplexity = float("inf")

        # 尝试在所有设备上计算困惑度的平均值，如果发生任何错误，就忽略。
        try:
            """
            - get_all_reduce_mean中调用了
              torch.distributed.all_reduce(perplexity, op=torch.distributed.ReduceOp.SUM)
              
            - 对所有进程、或者说GPU（因为通常情况下就是单个进程控制单个GPU）中的perplexity进行求和
            - 然后再除以全局进程数torch.distributed.get_world_size()得到平均的perplexity结果
            """
            # 如果这是一个分布式设置，该函数会计算所有device上的平均困惑度
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass

        # 返回最后的困惑度作为模型在给定评估数据集上的性能指标
        return perplexity

    # Train! 第7步：模型训练

    # 使用 print_rank_0 函数在主节点（global_rank为0的节点）打印开始训练的信息。
    print_rank_0("***** Running training *****", args.global_rank)

    # 在主节点打印在第0个epoch（训练开始前）进行模型评估的信息。
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)

    # 调用 evaluation 函数对模型进行评估，得到模型的困惑度。
    perplexity = evaluation(model, eval_dataloader)

    # 在主节点打印模型的困惑度。
    print_rank_0(f"ppl(perplexity): {perplexity}", args.global_rank)

    # 循环 args.num_train_epochs 轮进行训练。
    for epoch in range(args.num_train_epochs):
        # 在每个epoch开始时，在主节点(rank为0的进程中)打印开始新的训练周期的信息。
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)

        # 将模型设置为训练模式。
        gd.debuginfo(prj='ds_chat', info=f'Before ================')
        model.train()
        gd.debuginfo(prj='ds_chat', info=f'After ================')

        # 对于训练数据集中的每一个batch。
        for step, batch in enumerate(train_dataloader):
            # 将batch数据移到对应的设备上。
            batch = to_device(batch, device)  # torch函数

            logf = f'ph1_z{args.zero_stage}_forward_epoch{epoch:02}_step{step:04}'
            # if args.local_rank == 0:
            # gd.enable(info=logf)
            gd.emb_start(info=logf)

            # 将batch数据输入模型，进行前向计算；use_cache 是否使用缓存来加速计算
            outputs = model(**batch, use_cache=False)  # 触发引擎engine.py=def forward(self
            gd.debuginfo(prj='ds_chat', info=f'++forward_epoch{epoch:02}_step{step:04}_sep1++++++++')

            # 取出模型的输出中的loss。
            loss = outputs.loss

            #if args.print_loss:
            gd.debuginfo(prj='ds_chat', info=f"+++++Epoch: {epoch}, Step: {step}, "
                                             f"Rank: {torch.distributed.get_rank()}, "
                                             f"loss = {loss}")

            #if args.local_rank == 0:
            # gd.disable(info=logf)
            gd.emb_end(info=logf)

            logf = f'ph1_z{args.zero_stage}_backward_epoch{epoch:02}_step{step:04}'
            # if args.local_rank == 0:
            # gd.enable(info=logf)
            gd.emb_start(info=logf)

            # 调用模型的backward方法进行反向传播，计算损失函数关于模型参数的梯度。
            model.backward(loss) #触发引擎engine.py的backward=def backward(self

            gd.debuginfo(prj='ds_chat', info=f'+++forward_epoch{epoch:02}_step{step:04}_sep2++++++++')

            # 更新模型的参数。
            model.step() #触发引擎engine.py的backward=def step(self

            # if args.local_rank == 0:
            # gd.disable(info=logf)
            gd.emb_end(info=logf)

        # Evaluate perplexity on the validation set.
        # 在每个epoch结束后，在主节点打印开始评估的信息。
        print_rank_0(f"***** Evaluating perplexity, Epoch {epoch + 1}/{args.num_train_epochs} *****",
                     args.global_rank)

        # 对模型进行评估，计算在验证集上的困惑度
        perplexity = evaluation(model, eval_dataloader)
		
        # 在主节点打印模型的困惑度
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
		
        # 更新了模型的内部计时器，表示一个epoch已经完成
        model.tput_timer.update_epoch_count()

    # 第8步：训练结束后保存模型 如果设置了输出目录，进行以下操作。 
    if args.output_dir is not None:
        # 在主节点打印开始保存模型的信息。
        print_rank_0('saving the final model ...', args.global_rank)
		
        # 将模型中的LoRA层转换为标准的线性层，这样使得模型在保存后可以在没有LoRA层代码的环境中加载和使用
        model = convert_lora_to_linear_layer(model)
        gd.debuginfo(prj="ds_chat", info=f"ph1 convert_lora_to_linear_layer model {model}")

        if args.zero_stage == 2 or args.zero_stage == 3:
            mem_estimate_log(args=args, exstr='-ph1-3', model=model, num_gpus_per_node=2, num_nodes=1)

        # 如果是主节点，进行以下操作。 # 是否在主进程中
        if args.global_rank == 0:
            # 以HuggingFace格式保存模型和tokenizer
            save_hf_format(model, tokenizer, args)


        # ZeRO-3是一种内存优化策略，可以大大减少模型训练中所需的GPU内存，但同时也意味着模型的各部分在不同的GPU之间分布。
        if args.zero_stage == 3:
            logf = f'ph1_z{args.zero_stage}_save_zero_three_model'
            # if args.local_rank == 0:
            # gd.enable(info=logf)
            gd.emb_start(info=logf)

            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            # 使用特殊的保存函数保存模型。在Zero的第三阶段，每个GPU只有模型的一部分，所以需要特殊的保存函数。
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)
            #if args.local_rank == 0:
            # gd.disable(info=logf)
            gd.emb_end(info=logf)

if __name__ == "__main__":

    gd.debuginfo(prj='ds_chat', info=f'=================') # 不被计入
    gd.setIgnore(prj='ds', ignore=14)
    # 33 len('/home/amd00/yk_repo/ds/DeepSpeed/')
    # 14 len('/ds/DeepSpeed/')
    gd.setIgnore(prj='ds_chat', ignore=49)
    # 49 == len('/ds/DeepSpeedExamples/applications/DeepSpeed-Chat')
    # 69 == len('/home/amd00/yk_repo/ds/DeepSpeedExamples/applications/DeepSpeed-Chat/')

    gd.prjenable('ALL')  #打开项目flag

    gd.emb_mode(path=f'/log/_log_tmps_/', embedded_mode=True)

    main()

    gd.emb_mode(embedded_mode=False)

'''
ds_config is--1: {
'train_batch_size': 32, 
'train_micro_batch_size_per_gpu': 4, 
'steps_per_print': 10, 
'zero_optimization': {
    'stage': 2, 
    'offload_param': {'device': 'none'}, 
    'offload_optimizer': {'device': 'none'}, 
    'stage3_param_persistence_threshold': 10000.0, 
    'stage3_max_live_parameters': 30000000.0, 
    'stage3_prefetch_bucket_size': 30000000.0, 
    'memory_efficient_linear': False
    }, 
'fp16': {
    'enabled': True, 
    'loss_scale_window': 100
    }, 
'gradient_clipping': 1.0, 
'prescale_gradients': False, 
'wall_clock_breakdown': False, 
'hybrid_engine': {
    'enabled': False, 
    'max_out_tokens': 512, 
    'inference_tp_size': 1, 
    'release_inference_cache': False, 
    'pin_parameters': True, 
    'tp_gather_partition_size': 8
    }, 
'tensorboard': {
    'enabled': False, 
    'output_path': 'step1_tensorboard/ds_tensorboard_logs/', 
    'job_name': 'step1_model_tensorboard'
    }
}

ds_config['train_batch_size']is: 8
args.per_device_train_batch_size is: 4
'''

'''
model---4 : DeepSpeedEngine(
  (module): OPTForCausalLM(
    (model): OPTModel(
      (decoder): OPTDecoder(
        (embed_tokens): Embedding(50272, 768, padding_idx=1)
        (embed_positions): OPTLearnedPositionalEmbedding(2050, 768)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (layers): ModuleList(
          (0-11): 12 x OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): LinearLayer_LoRA(
                (lora_dropout): Identity()
              )
              (v_proj): LinearLayer_LoRA(
                (lora_dropout): Identity()
              )
              (q_proj): LinearLayer_LoRA(
                (lora_dropout): Identity()
              )
              (out_proj): LinearLayer_LoRA(
                (lora_dropout): Identity()
              )
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (fc1): LinearLayer_LoRA(
              (lora_dropout): Identity()
            )
            (fc2): LinearLayer_LoRA(
              (lora_dropout): Identity()
            )
            (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
    (lm_head): Linear(in_features=768, out_features=50272, bias=False)
  )
)
'''
 
'''
model---3 : DeepSpeedEngine(
  (module): OPTForCausalLM(
    (model): OPTModel(
      (decoder): OPTDecoder(
        (embed_tokens): Embedding(50272, 768, padding_idx=1)
        (embed_positions): OPTLearnedPositionalEmbedding(2050, 768)
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (layers): ModuleList(
          (0-11): 12 x OPTDecoderLayer(
            (self_attn): OPTAttention(
              (k_proj): LinearLayer_LoRA(
                (lora_dropout): Identity()
              )
              (v_proj): LinearLayer_LoRA(
                (lora_dropout): Identity()
              )
              (q_proj): LinearLayer_LoRA(
                (lora_dropout): Identity()
              )
              (out_proj): LinearLayer_LoRA(
                (lora_dropout): Identity()
              )
            )
            (activation_fn): ReLU()
            (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (fc1): LinearLayer_LoRA(
              (lora_dropout): Identity()
            )
            (fc2): LinearLayer_LoRA(
              (lora_dropout): Identity()
            )
            (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
    )
    (lm_head): Linear(in_features=768, out_features=50272, bias=False)
  )
)
optimizer---3 : <deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer object at 0x7f1469d45760>
lr_scheduler---3 : <torch.optim.lr_scheduler.LambdaLR object at 0x7f1492128040>
'''

'''
model---1 : OPTForCausalLM(
              (model): OPTModel(
                (decoder): OPTDecoder(
                  (embed_tokens): Embedding(50272, 768, padding_idx=1)
                  (embed_positions): OPTLearnedPositionalEmbedding(2050, 768)
                  (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (layers): ModuleList(
                    (0-11): 12 x OPTDecoderLayer(
                      (self_attn): OPTAttention(
                        (k_proj): Linear(in_features=768, out_features=768, bias=True)
                        (v_proj): Linear(in_features=768, out_features=768, bias=True)
                        (q_proj): Linear(in_features=768, out_features=768, bias=True)
                        (out_proj): Linear(in_features=768, out_features=768, bias=True)
                      )
                      (activation_fn): ReLU()
                      (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                      (fc1): Linear(in_features=768, out_features=3072, bias=True)
                      (fc2): Linear(in_features=3072, out_features=768, bias=True)
                      (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                    )
                  )
                )
              )
              (lm_head): Linear(in_features=768, out_features=50272, bias=False)
            )   
'''
'''
0--tokenizer : GPT2TokenizerFast(name_or_path='facebook/opt-125m', vocab_size=50265, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True)}, clean_up_tokenization_spaces=True)
1--tokenizer : GPT2TokenizerFast(name_or_path='facebook/opt-125m', vocab_size=50265, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': '</s>'}, clean_up_tokenization_spaces=True)

ds_config is--2: {
'train_batch_size': 8, 
'train_micro_batch_size_per_gpu': 4, 
'steps_per_print': 10, 
'zero_optimization': {
    'stage': 2, 
    'offload_param': {'device': 'none'}, 
    'offload_optimizer': {'device': 'none'}, 
    'stage3_param_persistence_threshold': 10000.0, 
    'stage3_max_live_parameters': 30000000.0, 
    'stage3_prefetch_bucket_size': 30000000.0, 
    'memory_efficient_linear': False
    }, 
'fp16': {
    'enabled': True, 
    'loss_scale_window': 100
    }, 
'gradient_clipping': 1.0, 
'prescale_gradients': False, 
'wall_clock_breakdown': False, 
'hybrid_engine': {
    'enabled': False, 
    'max_out_tokens': 512, 
    'inference_tp_size': 1, 
    'release_inference_cache': False,
    'pin_parameters': True, 
    'tp_gather_partition_size': 8
    }, 
'tensorboard': {
    'enabled': False, 
    'output_path': 'step1_tensorboard/ds_tensorboard_logs/',
    'job_name': 'step1_model_tensorboard'
    }
}
'''
'''
train_dataset : <utils.data.data_utils.PromptDataset object at 0x7f147c395c10>
eval_dataset : <utils.data.data_utils.PromptDataset object at 0x7f14920c5af0>
train_sampler : <torch.utils.data.distributed.DistributedSampler object at 0x7f147c456ee0>
eval_sampler : <torch.utils.data.distributed.DistributedSampler object at 0x7f14753f2580>
train_dataloader : <torch.utils.data.dataloader.DataLoader object at 0x7f14753f2460>
eval_dataloader : <torch.utils.data.dataloader.DataLoader object at 0x7f149210d190>
'''

'''
Parameter containing Pytorch   https://zhuanlan.zhihu.com/p/119305088
The PyTorch parameter is a layer made up of nn or a module. 
A parameter that is assigned as an attribute inside a custom model is registered as a model parameter and 
is thus returned by the caller model. parameters(). 
We can say that a Parameter is a wrapper over Variables that are formed.

optimizer_grouped_parameters : [{'params': [Parameter containing:
tensor([[ 0.1150, -0.1438,  0.0555,  ...,  0.2146,  0.0833,  0.0669],
...
        [ 0.1156, -0.1437,  0.0577,  ...,  0.2139,  0.0833,  0.0650]],
       requires_grad=True), Parameter containing:
tensor([[ 1.8272e-03,  9.0599e-04,  4.4289e-03,  ...,  1.6693e-02,
          1.7462e-03, -4.9057e-03],

'''

'''
optimizer : FusedAdam (
            Parameter Group 0
                betas: (0.9, 0.95)
                bias_correction: True
                eps: 1e-08
                lr: 9.65e-06
                weight_decay: 0.0

            Parameter Group 1
                betas: (0.9, 0.95)
                bias_correction: True
                eps: 1e-08
                lr: 0.0005
                weight_decay: 0.0

            Parameter Group 2
                betas: (0.9, 0.95)
                bias_correction: True
                eps: 1e-08
                lr: 9.65e-06
                weight_decay: 0.0
            )
'''