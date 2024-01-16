#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys

# pid = os.getpid()

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from pydebug import gd, infoTensor

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer, mem_estimate_log
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.utils import mem_estimate_log

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    # 1 训练数据集的路径
    parser.add_argument('--data_path',
                        nargs='*', # 可以接受任意数量的值，这些值将被收集到一个列表中  ？？？
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')

    # 2 数据集的切分比例
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')

    # 3 存储与数据相关（如shuffle index）的文件的位置
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')

    # 4 预训练模型的路径 或者来自 huggingface.co/models的 model identifier
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    # 5 起始位置的填充数量， 其他模型没有，仅仅是opt有一个固定为1的起始位置的填充!!
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )

    # 6 每个设备上的训练批次大小
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )

    # 7 每个设备上的评估批次大小
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
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
        default=5e-5,
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

    # 13 学习率调度器类型
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
    '''
    warm-up是针对学习率learning rate优化的一种策略，
    https://blog.csdn.net/wyf2017/article/details/123956875
    主要过程是:在预热期间，学习率从0线性（也可非线性）增加到优化器中的初始预设lr，
    之后使其学习率从优化器中的初始lr线性降低到0
    '''
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

    # 18 是否启用梯度checkpoint
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')

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
                        default=0,
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
                        default="step2_tensorboard")

    # 这一行将DeepSpeed的配置参数添加到解析器中。
    parser = deepspeed.add_config_arguments(parser)

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


def main():
    # 第1步：超参数配 解析命令行参数。
    args = parse_args()

    # 如果local_rank为-1，说明不在分布式训练环境下，设备设置为"cuda"；
    # 否则根据args.local_rank设置对应的cuda设备，并初始化分布式训练。
    if args.local_rank == -1:
        # 单机版的CUDA
        device = torch.device("cuda")
    else:
        # 分布式训练
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(ba  ckend='nccl')

        # 初始化分布式训练环境
        deepspeed.init_distributed()

    # 获取当前运行设备在分布式训练环境中的全局rank
    args.global_rank = torch.distributed.get_rank()
    gd.debuginfo(prj="ds_chat", info=f"args.global_rank: {args.global_rank}")
	
    # 根据输入参数返回一个训练数据集的配置字典
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step2_model")
    gd.debuginfo(prj="ds_chat", info=f"ph2 ds_config train---1, {ds_config}") #一直打开
									
    # micro_batch训练是一种分布式训练技术，可以将一个大批次的数据分解成多个小批次，以适应GPU的内存限制
    # 在配置中设置训练时每个GPU的微批次大小和总的批次大小。
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size

    # 每个训练步骤中处理的数据总量
    # gradient_accumulation(梯度积累)是另一种应对内存限制的技术，它会在多个步骤中积累梯度，然后再一次性更新模型参数。
    # torch.distributed.get_world_size() 在分布式训练环境中获取节点（设备）数量
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    gd.debuginfo(prj="ds_chat", info=f"ph2 ds_config train---2, {ds_config}") #一直打开

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    # 确保所有分布式进程都达到了这个阻塞点，
    # 然后再继续执行后面的代码，以避免出现某些进程快于其他进程的情况。
    torch.distributed.barrier()
	
    # 表示使用优化过的、速度更快的tokenizer。
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    gd.debuginfo(prj="ds_chat", info=f"ph2 tokenizer---0, {tokenizer}")
	
    # 模型将认为这些填充部分是句子的结束。
    tokenizer.pad_token = tokenizer.eos_token
	
    # make sure tokenizer is right pad in our logic
	# 在序列的右侧（末尾）添加填充符号。
    tokenizer.padding_side = 'right'

    gd.debuginfo(prj="ds_chat", info=f"ph2 tokenizer---1, {tokenizer}")
	
    """
    rm_model调用了create_critic_model进行载入
    默认情况下rm_model是不启用dropout的
    """
    # 第2步：创建reward模型
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   disable_dropout=args.disable_dropout)

    gd.debuginfo(prj="ds_chat", info=f"s2 create_critic_model rm_model, {rm_model}")
    if args.zero_stage == 2 or args.zero_stage == 3:
        mem_estimate_log(args=args, exstr = '-ph2-0', model=rm_model, num_gpus_per_node=2, num_nodes=1)

    if args.lora_dim > 0:
        # 将模型中指定的线性层转换为LoRA层
        # lora_module_name指定了要转换的模块的名称  lora_dim指定了LoRA的维度
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        gd.debuginfo(prj="ds_chat", info=f"s2 convert_linear_layer_to_lora rm_model, {rm_model}")
        if args.zero_stage == 2 or args.zero_stage == 3:
            mem_estimate_log(args=args, exstr = '-ph2-1', model=rm_model, num_gpus_per_node=2, num_nodes=1)

        if args.only_optimize_lora:
            # 将模型中非LoRA层的参数的requires_grad属性设为False，在训练过程中只有LoRA层的参数会被更新。
            rm_model = only_optimize_lora_parameters(rm_model)
            gd.debuginfo(prj="ds_chat", info=f"s2 only_optimize_lora_parameters rm_model, {rm_model}")
            if args.zero_stage == 2 or args.zero_stage == 3:
                mem_estimate_log(args=args, exstr = '-ph2-2', model=rm_model, num_gpus_per_node=2, num_nodes=1)

    # 第3步：准备数据集（训练集和验证集）Prepare the data
    # 创建数据集和数据加载器：包括训练集和验证集，以及对应的采样器和数据加载器。
    train_phase = 2

    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)
    gd.debuginfo(prj="ds_chat", info=f"train_dataset, {train_dataset}")
    gd.debuginfo(prj="ds_chat", info=f"eval_dataset, {eval_dataset}")
    gd.debuginfo(prj="ds_chat", info=f"len of train_dataset, {len(train_dataset)}")
    gd.debuginfo(prj="ds_chat", info=f"len of eval_dataset, {len(eval_dataset)}")

    # DataLoaders creation:
    """
    2.3.2 DataCollator及RM所需输入形式
    phase2使用的数据整理器data_collator为DataCollatorReward()，
    本阶段取出的单个样本example实际上是一个chosen-rejected数据对（见下方代码块），
    即1个大小为batch_size的batch取出了batch_size个数据对，
    data_collator将把数据对拆成chosen_sentence和reject_sentence（example一分为二），
    因此实际上1个batch真正输入模型的数据量大小应当为“batch_size * 2”。
    """
    # phase2使用的data_collator为DataCollatorReward()
	# 2. 将批次数据整理成模型需要的形式
    data_collator = DataCollatorReward()
    gd.debuginfo(prj="ds_chat", info=f"data_collator , {data_collator}")

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

    gd.debuginfo(prj="ds_chat", info=f"train_sampler , {train_sampler}")
    gd.debuginfo(prj="ds_chat", info=f"eval_sampler , {eval_sampler}")

    #default_data_collator 作用是将一批数据进行整合，使得它们可以整齐地堆叠在一起。

    # 创建用于训练集的数据加载器
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset) #yknote和step1不一样？？？？
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    gd.debuginfo(prj="ds_chat", info=f"train_dataloader , {train_dataloader}")
    gd.debuginfo(prj="ds_chat", info=f"eval_sampler , {eval_sampler}")
    gd.debuginfo(prj="ds_chat", info=f"eval_dataloader , {eval_dataloader}")

    '''
    2.3.4 phase2的指标评估
    DeepSpeed-Chat在phase2中使用的评估指标为排序正确的accuracy，主要过程为：
    
        1将数对chosen-rejected数据对（过程中被data_collator拆分为chosen_sentence和reject_sentence）输入RM中进行推理，
        得到各个sentence的分值；
        
        2将同属一个prompt的chosen_sentence得分与reject_sentence得分进行比较，
        当chosen_sentence得分大于reject_sentence得分时，即为“正确预测”，否则为“错误预测”；
        
        3统计正确预测的结果，计算accuracy作为评估指标。
        
        4此外评估过程中还将统计平均的chosen_sentence分值“scores”供参考。
        '''

    # Split weights in two groups, one with weight decay and the other not.
	# 第4步：将模型参数分组、创建优化器 和 学习率调度器
	# 1. 将模型的参数分为两组，一组应用权重衰减，另一组不应用
    # 权重衰减是防止模型过拟合的一种策略，通常只对模型的权重参数应用。
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(rm_model, args.weight_decay)
    gd.debuginfo(prj="ds_chat", info=f"optimizer_grouped_parameters , {optimizer_grouped_parameters}")

    # 根据是否使用DeepSpeed的CPU offload功能来选择优化器，优化器定义了如何更新模型的参数以最小化损失函数。

    # DeepSpeedCPUAdam : 配合DeepSpeed的CPU offload功能设计的，
    #          CPU offload可以将模型参数、优化器状态和梯度数据在CPU和GPU之间进行切换，以减轻GPU的内存压力。

    # FusedAdam : 它将一些计算操作融合在一起（fused），以减少计算时间和内存使用量。
    #          FusedAdam主要是为了提高在GPU上的运算效率。
	
    # 选择优化器类型，如果启用了梯度Offload，使用DeepSpeedCPUAdam，否则使用FusedAdam。
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
	
    gd.debuginfo(prj="ds_chat", info=f"AdamOptimizer , {AdamOptimizer}")
    # AdamOptimizer : <class 'deepspeed.ops.adam.fused_adam.FusedAdam'>
    # 优化器被初始化时，指定了模型参数、学习率和优化器的betas参数。
    # 创建优化器
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    gd.debuginfo(prj="ds_chat", info=f"optimizer , {optimizer}")

    # 计算每个epoch需要进行的更新步数，等于训练数据集大小除以梯度累积步数（对结果向上取整）
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    gd.debuginfo(prj="ds_chat", info=f"num_update_steps_per_epoch , {num_update_steps_per_epoch}")

    # 创建学习率调度器，学习率调度器可以动态地在训练过程中改变学习率以得到最优的学习效果。
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type, # 调度器的类型
        optimizer=optimizer, # 优化器，在每个训练步骤调整其学习率
        num_warmup_steps=args.num_warmup_steps, # 预热步骤数，在训练开始的一段时间内，
                                                # 学习率从0线性增加到预设的初始学习率，预热过程有助于模型的稳定训练。
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch, # 总的训练步骤数
    )
    gd.debuginfo(prj="ds_chat", info=f"lr_scheduler---2 , {lr_scheduler}")
    # lr_scheduler---2 : <torch.optim.lr_scheduler.LambdaLR object at 0x7f1c900786a0>
	
    '''--- 知识点补充 ---
    调度器类型有哪些？
    LINEAR：线性调度器，学习率将在训练过程中线性递减。
    COSINE：余弦调度器，学习率将在训练过程中按照余弦函数递减。
    COSINE_WITH_RESTARTS：带重启的余弦调度器，学习率将按照余弦函数递减，但在每个周期结束时重启。
    POLYNOMIAL：多项式调度器，学习率将在训练过程中按照多项式函数递减。
    CONSTANT：常数调度器，学习率在训练过程中保持不变。
    CONSTANT_WITH_WARMUP：带预热的常数调度器，学习率在一开始的一段时间内线性增加，然后保持不变。
    '''
    # 第5步：deepspeed初始化，创建模型、优化器、学习率调度器
    # if args.local_rank == 0:
    logf = f'ph2_z{args.zero_stage}_deepspeed.initialize'
    # gd.enable(info=logf)
    gd.emb_start(info=logf)

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model, # 模型
        optimizer=optimizer, # 优化器
        args=args, # 参数对象
        config=ds_config, # DeepSpeed的配置信息
        lr_scheduler=lr_scheduler, # 学习率调度器
        dist_init_required=True # # 需要进行分布式训练的初始化
    )
    gd.debuginfo(prj="ds_chat", info=f"rm_model---4 , {rm_model}")
    gd.debuginfo(prj="ds_chat", info=f"optimizer---4 , {optimizer}")
    gd.debuginfo(prj="ds_chat", info=f"lr_scheduler---4 , {lr_scheduler}")

    #if args.local_rank == 0:
    # gd.disable(info=logf)
    gd.emb_end(info=logf)

    if args.gradient_checkpointing:
        # 在模型中启用梯度检查点
        # # 梯度检查点是一种可以减少训练过程中内存使用的技术，通过牺牲一部分计算效率来换取内存占用的降低，会增加一些计算时间。
        rm_model.gradient_checkpointing_enable()

    # 第6步：模型验证   yknote---改变位置
    def evaluation_reward(model, eval_dataloader):
        # 评估模式
        model.eval()
        # 统计预测（赋分）正确的结果即chosen_reward > rejected_reward的结果数  # 预测正确的数量
        correct_predictions = 0

        # 统计预测总数 # 总预测数量
        total_predictions = 0
        scores = 0
        gd.debuginfo(prj="ds_chat", info="len(eval_dataloader): {len(eval_dataloader)}")
        for step, batch in enumerate(eval_dataloader):
            # gd.debuginfo(prj="ds_chat", info=f"batch---C is, {batch}")
            gd.debuginfo(prj="ds_chat", info=f"T batch['input_ids']--C, {infoTensor(batch['input_ids'])}")
            # #only ph2 T batch['input_ids']--C: _Size([16, 128])_int64_cpu_
            gd.debuginfo(prj="ds_chat", info=f"T batch['attention_mask']--C, {infoTensor(batch['attention_mask'])}")
            # #pnly ph2 T batch['attention_mask']--C: _Size([16, 128])_int64_cpu_

            batch = to_device(batch, device)

            # 禁用梯度计算
            with torch.no_grad():
                """
                outputs: {'loss':tensor(), 
                            'chosen_mean_scores':tensor(bs,), 
                            'rejected_mean_scores':tensor(bs,)}
                """
                # 前向传播
                outputs = model(**batch)
                #gd.debuginfo(prj="ds_chat", info=f"outputs--C, {outputs}")
                gd.debuginfo(prj="ds_chat", info=f"T outputs['loss']--A, {infoTensor(outputs['loss'])}")
                gd.debuginfo(prj="ds_chat", info=f"T outputs['chosen_mean_scores']--A, {infoTensor(outputs['chosen_mean_scores'])}")
                gd.debuginfo(prj="ds_chat", info=f"T outputs['rejected_mean_scores']--A, {infoTensor(outputs['rejected_mean_scores'])}")

            # 获取预测得分
            # chosen.shape: (bs,)
            chosen = outputs["chosen_mean_scores"]

            # rejected.shape: (bs,)
            rejected = outputs["rejected_mean_scores"]

            # gd.debuginfo(prj="ds_chat", info=f"chosen--C: {chosen}")
            # gd.debuginfo(prj="ds_chat", info=f"rejected--C: {rejected}")
            gd.debuginfo(prj="ds_chat", info=f"T chosen---C, {infoTensor(chosen)}")
            gd.debuginfo(prj="ds_chat", info=f"T rejected---C, {infoTensor(rejected)}")
            ''' only ph2
            chosen--C tensor([-0.4812, -0.0686, -0.5049, -0.2944, -0.4268, -0.7700, -0.4253, -0.3943],
                   device='cuda:0', dtype=torch.float16)
            rejected--C tensor([-0.4812, -0.0686, -0.5049, -0.2944, -0.4268, -0.7700, -0.4253, -0.3943],
                   device='cuda:1', dtype=torch.float16)
            T chosen---C: _Size([8])_float16_cuda:1_
            T rejected---C: _Size([8])_float16_cuda:1_
            '''

            # 赋分正确"即为chosen分值大于rejected分值
            # 更新正确预测数量和总预测数量
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]

            # only ph2
            gd.debuginfo(prj="ds_chat", info=f"correct_predictions--C: {infoTensor(correct_predictions)}")
            # correct_predictions--C _Size([])_int64_cuda:0_  only ph2

            gd.debuginfo(prj="ds_chat", info=f"total_predictions--C: {total_predictions}")  #total_predictions--C 672

            # 累加每个step的平均chosen分值
            # 计算并累计得分
            scores += outputs["chosen_mean_scores"].mean().float()

            if step == 99:  # For faster evaluation and debugging
                break

        # 计算准确率acc指标
        acc = correct_predictions / total_predictions

        # 计算平均得分, 当前step的平均chosen分值
        scores = scores / (step + 1)
        try:
            # 多进程结果求和求平均  # 使用分布式计算的平均准确率和得分
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
        except:
            pass
        gd.debuginfo(prj="ds_chat", info=f"scores: {scores}")
        gd.debuginfo(prj="ds_chat", info=f"acc: {acc}")
        return scores, acc


    # Train!  # 第7步：模型训练
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)

    # 在训练集上评估模型的奖励值
    #if args.local_rank == 0:
    logf = f'ph2_z{args.zero_stage}_evaluation_reward_B'
    # gd.enable(info=logf)
    gd.emb_start(info=logf)

    reward_score, acc = evaluation_reward(rm_model, eval_dataloader)

    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
        args.global_rank)

    #if args.local_rank == 0:
    # gd.disable(info=logf)
    gd.emb_end(info=logf)

    # 模型训练
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)

        # logf = f'ph2_z{args.zero_stage}_rm_model.train model'
        # if args.local_rank == 0:
        #     gd.enable(info=logf)
        # 训练模式
        gd.debuginfo(prj='ds_chat', info=f'Before rm_model.train()++++++++')
        rm_model.train()
        gd.debuginfo(prj='ds_chat', info=f'After rm_model.train() ++++++++')

        mean_loss = 0
        # if args.local_rank == 0:  # ph2-z01234 log录得为空！
        #     gd.disable(info=logf)

        for step, batch in enumerate(train_dataloader):
            logf = f'ph2_z{args.zero_stage}_rm_model_epoch{epoch:02}_step{step:04}'
            #if args.local_rank == 0:
            # gd.enable(info=logf)
            gd.emb_start(info=logf)

            gd.debuginfo(prj='ds_chat', info=f'1-batch={batch}')
            batch = to_device(batch, device)
            gd.debuginfo(prj='ds_chat', info=f'2-batch={batch}')


            # 将批数据输入模型并获取输出
            outputs = rm_model(**batch, use_cache=False)
            gd.debuginfo(prj='ds_chat', info=f'+++_epoch{epoch:02}_step{step:04}_sep1++++++++')
			
            # 从模型输出中提取损失
            loss = outputs["loss"]
            # print容易引起打错过多的显示错位！！！
            # print_rank_0("batch :", batch)
            # print_rank_0("outputs :", outputs)
            # print_rank_0("loss :", loss)

            gd.debuginfo(prj="ds_chat", info=f"T batch['input_ids']--D, {infoTensor(batch['input_ids'])}")
            gd.debuginfo(prj="ds_chat", info=f"T batch['attention_mask']--D, {infoTensor(batch['attention_mask'])}")
            gd.debuginfo(prj="ds_chat", info=f"T outputs['loss']--D, {infoTensor(outputs['loss'])}")
            gd.debuginfo(prj="ds_chat", info=f"T outputs['chosen_mean_scores']--D, {infoTensor(outputs['chosen_mean_scores'])}")
            gd.debuginfo(prj="ds_chat", info=f"T outputs['rejected_mean_scores']--D, {infoTensor(outputs['rejected_mean_scores'])}")
            gd.debuginfo(prj="ds_chat", info=f"T loss--D, {infoTensor(loss)}")
            '''
            T batch['input_ids']--D: _Size([16, 128])_int64_cuda:1_
            T batch['attention_mask']--D: _Size([16, 128])_int64_cuda:1_
            T outputs['loss']--D: _Size([])_float16_cuda:1_
            T outputs['chosen_mean_scores']--D: _Size([8])_float16_cuda:1_
            T outputs['rejected_mean_scores']--D: _Size([8])_float16_cuda:1_
            T loss--D: _Size([])_float16_cuda:1_
            '''

            gd.debuginfo(prj='ds_chat', info=f'+++_epoch{epoch:02}_step{step:04}_sep2++++++++')
            # 计算损失的梯度
            rm_model.backward(loss)

            gd.debuginfo(prj='ds_chat', info=f'+++_epoch{epoch:02}_step{step:04}_sep3++++++++')

            # 用计算的梯度更新模型的权重
            rm_model.step()
            gd.debuginfo(prj='ds_chat', info=f'+++_epoch{epoch:02}_step{step:04}_sep4++++++++')





            # 计算所有批次的平均损失
            mean_loss += loss.item()
            gd.debuginfo(prj='ds_chat', info=f'mean_loss={mean_loss}')

            #if args.local_rank == 0:
            # gd.disable(info=logf)
            gd.emb_end(info=logf)


        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)

        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)

        # 在每个epoch结束后，模型在验证数据集上进行评估。
        logf = f'ph2_z{args.zero_stage}_evaluation_reward_A'
        #if args.local_rank == 0:
        # gd.enable(info=logf)
        gd.emb_start(info=logf)

        reward_score, acc = evaluation_reward(rm_model, eval_dataloader)

        #if args.local_rank == 0:
        # gd.disable(info=logf)
        gd.emb_end(info=logf)

        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
            args.global_rank)

        # 更新模型的吞吐量计时器以记录完成的epoch数
        rm_model.tput_timer.update_epoch_count()

    # 第8步：训练结束后保存模型 如果设置了输出目录，进行以下操作。 
    if args.output_dir is not None:
        # 在主节点打印开始保存模型的信息。
        print_rank_0('saving model ...', args.global_rank)
        # 将模型中的LoRA层转换为全连接层，这样使得模型在保存后可以在没有LoRA层代码的环境中加载和使用
        rm_model = convert_lora_to_linear_layer(rm_model)
        gd.debuginfo(prj="ds_chat", info=f"ph2 convert_lora_to_linear_layer model, {rm_model}")

        mem_estimate_log(args=args, exstr = '-ph2-3', model=rm_model, num_gpus_per_node=2, num_nodes=1)

        # 如果是主节点，进行以下操作。 # 是否在主进程中
        if args.global_rank == 0:
            # 以HuggingFace格式保存模型和tokenizer
            save_hf_format(rm_model, tokenizer, args)

        # ZeRO-3是一种内存优化策略，可以大大减少模型训练中所需的GPU内存，但同时也意味着模型的各部分在不同的GPU之间分布。
        if args.zero_stage == 3:
            logf = f'ph2_z{args.zero_stage}_save_zero_three_model'
            # if args.local_rank == 0:
            # gd.enable(info=logf)
            gd.emb_start(info=logf)

            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            # 使用特殊的保存函数保存模型。在Zero的第三阶段，每个GPU只有模型的一部分，所以需要特殊的保存函数。
            save_zero_three_model(rm_model,
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
ds_config--2 : {
'train_batch_size': 16, 
'train_micro_batch_size_per_gpu': 8, 
'steps_per_print': 10, 
'zero_optimization': {
    'stage': 0, 
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
    'output_path': 'step2_tensorboard/ds_tensorboard_logs/', 
    'job_name': 'step2_model_tensorboard'
    }
}
'''

'''

rm_model---1 : RewardModel(
  (v_head): Linear(in_features=768, out_features=1, bias=False)
  (rwtranrsformer): OPTModel(
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
)
'''
'''
rm_model---5 :  DeepSpeedEngine(
      (module): RewardModel(
        (v_head): Linear(in_features=768, out_features=1, bias=False)
        (rwtranrsformer): OPTModel(
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
      )
    )
'''

'''
batch : {'input_ids': tensor([[    2, 50118, 50118,  ...,     2,     2,     2],
        ...,
        [    2, 50118, 50118,  ...,     8,    24,    17]], device='cuda:0'),
         'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 1, 1, 1]], device='cuda:0')}
        
outputs : {
'loss': tensor(0.7422, device='cuda:0', dtype=torch.float16, grad_fn=<DivBackward0>),
'chosen_mean_scores': tensor([-0.4436,  0.2378,  0.6147,  0.7080,  0.5176, -0.6152,  0.3547, -0.4399], device='cuda:0', dtype=torch.float16, grad_fn=<StackBackward0>),
'rejected_mean_scores': tensor([-1.0381,  0.2378,  0.6147,  0.7080,  0.5176, -0.1653,  0.3547, -0.4399], device='cuda:0', dtype=torch.float16, grad_fn=<StackBackward0>)}
       
loss : tensor(0.7334, device='cuda:1', dtype=torch.float16, grad_fn=<DivBackward0>)
'''


'''
rm_model---4 : DeepSpeedEngine(
  (module): RewardModel(
    (v_head): Linear(in_features=768, out_features=1, bias=False)
    (rwtranrsformer): OPTModel(
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
  )
)
    
optimizer---4 : FusedAdam (
Parameter Group 0
    betas: (0.9, 0.95)
    bias_correction: True
    eps: 1e-08
    initial_lr: 5e-05
    lr: 5e-05
    step: 0
    weight_decay: 0.1

Parameter Group 1
    betas: (0.9, 0.95)
    bias_correction: True
    eps: 1e-08
    initial_lr: 5e-05
    lr: 5e-05
    step: 0
    weight_decay: 0.0
)

lr_scheduler---4 : <torch.optim.lr_scheduler.LambdaLR object at 0x7f1c900786a0>

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

'''
outputs--C {'loss': tensor(0.7271, device='cuda:0', dtype=torch.float16), 
        'chosen_mean_scores': tensor([-0.4062,  0.5601, -0.3333, -0.2805, -0.9033, -1.0156, -0.5679, -1.1670],
                    device='cuda:0', dtype=torch.float16), 
        'rejected_mean_scores': tensor([-0.4143,  0.5601,  0.6343, -0.2805, -0.9736, -1.0156, -0.5679,  0.4614],
                    device='cuda:0', dtype=torch.float16)}
T outputs['loss']--A: _Size([])_float16_cuda:0_
T outputs['chosen_mean_scores']--A: _Size([8])_float16_cuda:0_
T outputs['rejected_mean_scores']--A: _Size([8])_float16_cuda:0_
'''

'''
batch---C is: 
{'input_ids': tensor([[    2, 50118, 50118,  ...,   533,     7, 28616],
    ...,
    [    2, 50118, 50118,  ...,  1328,     5,   183]]), 
'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
...
    [1, 1, 1,  ..., 1, 1, 1]])}
'''

'''
train_dataset : <utils.data.data_utils.PromptDataset object at 0x7f99ca0ea3a0>
eval_dataset : <utils.data.data_utils.PromptDataset object at 0x7f99957c5ca0>
data_collator : <utils.data.data_utils.DataCollatorReward object at 0x7f99f80e6d00>
train_sampler : <torch.utils.data.distributed.DistributedSampler object at 0x7f99ce086130>
eval_sampler : <torch.utils.data.distributed.DistributedSampler object at 0x7f99f80bf310>
train_dataloader : <torch.utils.data.dataloader.DataLoader object at 0x7f99a0315070>
eval_sampler : <torch.utils.data.sampler.SequentialSampler object at 0x7f99a0315130>
eval_dataloader : <torch.utils.data.dataloader.DataLoader object at 0x7f99f80bf310>
'''