#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""
import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine

from pydebug import gd, infoTensor

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer
from utils.module.lora import convert_lora_to_linear_layer

writer = None


def parse_args():
    global writer

    parser = argparse.ArgumentParser(
        description="(Step 3) RLHF training arguments")

    # 1. 训练数据集的路径
    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )

    # 2. 数据分割比例
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60% of data for phase 1, 20% for phase 2 and 20% for phase 3.'
    )

    # 3. 数据相关文件的存储路径
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
	
    # 4. 无监督数据集  ===TBD找例子！
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
		
    # 5. 无监督数据集配置名称
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
	
    # 6.调整训练过程中的无监督学习部分的权重
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
						
    # 7. actor模型名称或路径
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
		
    # 8. critic模型名称或路径
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
		
    # 9. 起始位padding的数量
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
	
    # 10. train batch size
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
	
    # 11. min train batch size
    parser.add_argument(
        "--per_device_mini_train_batch_size",
        type=int,
        default=16,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
	
    # 12. yknote？？？就是batch size
    parser.add_argument("--generation_batch_numbers",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
						
    # 13. 控制对生成的数据运行多少个PPO（Proximal Policy Optimization，近端策略优化）训练周期
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
		
    # 14. 最大prompt序列长度
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
						
    # 15. 最大answer序列长度
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
						
    # 16. actor学习率
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
	
    # 17. critic学习率
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )

    # 18. actor权重衰减
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")

    # 19. critic权重衰减
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    # 20. 训练epochs
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")

    # 21. 学习率调度器类型
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

    # 22. 梯度累计步骤数
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")

    # 23. warmup步骤数
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")

    # 24. 输出路径
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")

    # 25. 随机种子
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")

    # 26. 预处理线程数
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    # 27.当前的进程在所有分布式进程中的位置（排名）
    # 这对于在多GPU环境下进行分布式训练时来说是非常重要的，
    # 如果--local_rank的值为-1，那么程序会认为当前环境不是分布式训练环境，将执行单机训练代码。
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # 28.是否开启DeepSpeed的混合引擎
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )

    # 29.在生成过程中取消锁定（unpin）actor（生成模型）的参数
    # 开启此选项可能会使生成速度变慢，但可以减少内存的使用。
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )

    # 30.释放用于推理的内存缓存
    # 开启此选项可能会使生成准备阶段的速度变慢，但可能会通过使用更大的batch size来提高端到端的吞吐量。
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. \
        This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )

    # 31.用于推理优化的张量并行度
    # 如果命令行中指定了该参数，那么就会使用相应的值作为并行度，使用此功能时必须启用混合引擎。
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )

    # 32.用于在混合引擎内进行张量并行（TP）划分的粒度
    # 如果命令行中指定了该参数，那么就会使用相应的值作为划分粒度，使用此功能时必须启用混合引擎。
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )

    # 33.启用ZeRO Offload 技术
    # 它将部分模型参数和优化器状态在CPU和GPU之间交换，以降低GPU内存的占用。
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')

    # 34.为reference模型启用ZeRO Offload技术
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')

    # 35. actor模型: zero优化阶段
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')

    # 36.critic模型: zero优化阶段
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')

    # 37.启用梯度检查点（gradient checkpointing）技术用于Actor模型
    # 梯度检查点技术是一种内存优化策略，它通过减少存储在训练过程中的激活值来节省内存，
    # 但这会增加计算的复杂性，因为需要重新计算这些激活值。
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')

    # 38.启用梯度检查点（gradient checkpointing）技术用于Critic模型
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')

    # 39.用于禁用Actor模型的dropout
    # Dropout是一种正则化技术，它通过随机关闭一部分神经元来防止模型过拟合。
    parser.add_argument('--disable_actor_dropout',
                        action='store_true',
                        help='Disable the dropout of the actor model.')
    # 40.与actor模型作用一样
    parser.add_argument('--disable_critic_dropout',
                        action='store_true',
                        help='Disable the dropout of the critical model.')

    ## LoRA for efficient training setting

    # 41.设置Actor模型的LoRA（Low Rank Adaptation）维度
    # LoRA是一种新的训练技术，它可以有效地适应新任务，同时保持预训练模型的参数不变。
    # LoRA通过在原有的模型参数上添加一组低秩参数来实现这个目标
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")

    # 42.指定在Actor模型中使用LoRA的模块名称
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")

    # 43.41.设置Critic模型的LoRA（Low Rank Adaptation）维度
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")

    # 44.指定在Critic模型中使用LoRA的模块名称
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    # 45.仅优化lora
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')

    ## Make EMA as an optional feature

    # 46.是否为模型使用指数移动平均（EMA）检查点
    # 指数移动平均（EMA）是一种常用的技术，用于平滑时间序列数据，特别是在处理噪音数据时。
    # EMA可以用于维护模型的参数，它会记录模型在训练过程中参数的移动平均，可以稳定训练过程并可能提高模型的泛化能力。
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')

    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')

    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step3_tensorboard")

    ## Actor/critic model overflow alignment
    parser.add_argument(
        '--align_overflow',
        action='store_true',
        help='Align loss scale overflow between actor and critic')

    ## Print actor model answers during training
    parser.add_argument('--print_answers',
                        action='store_true',
                        help='Print prompt and answers during training')

    parser = deepspeed.add_config_arguments(parser)
    gd.debuginfo(prj="ds_chat", info=f"ph3 parser is:, {parser}")

    args = parser.parse_args()
    gd.debuginfo(prj="ds_chat", info=f"ph3 args is:, {args}")

    if args.enable_tensorboard:
        print(f"Tensorboard logs going to: {args.tensorboard_path}/step3_tensorboard_logs")
        writer = SummaryWriter(
            f"{args.tensorboard_path}/step3_tensorboard_logs")

    # Validate settings
    # 当同时启用LoRa和梯度检查点时，可能会产生冲突，导致模型的训练效果受到影响，这两个功能不能同时启用。
    if (args.actor_gradient_checkpointing
            and args.actor_lora_dim > 0) or (args.critic_gradient_checkpointing
                                             and args.critic_lora_dim > 0):
        # 如果only_optimize_lora参数被启用，也就是说只优化LoRa的部分权重，
        # 这与梯度检查点也是冲突的，因为梯度检查点需要对所有的权重进行操作，如果只优化LoRa的部分权重，就不能正确地执行梯度检查点。
        assert (
            not args.only_optimize_lora
        ), "--{actor,critic}_gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    # inference_tp_size用于优化推理的张量并行度，
    # 当这个值大于1时，就意味着需要使用张量并行技术进行推理。
    if args.inference_tp_size > 1:
        # 要使用张量并行技术，必须在模型的分布式训练中启用ZeRO第三阶段
        # 第三阶段（ZeRO-3）可以同时减少GPU上的模型状态、优化器状态和梯度的存储量，从而允许训练更大的模型。
        assert (args.actor_zero_stage == 3), \
            "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    return args

# 3.3.1.3 无监督数据集的读取
# 无监督数据集主要是进行了分块处理，将无监督语料全部拼接起来得到一个极长的序列，
# 使用max_seq_len大小的滑窗对长序列进行分块，每个分块将作为1条无监督数据。
def create_datasets(args, tokenizer, train_phase=3):
    # 是否启用了无监督训练
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    """
    获取Dataset和实例化dataloader
    """

    """
    返回读取到的prompt数据，
    该数据为经由tokenizer处理（tokenize但未padding），
    且本地存储后的 input_ids 和 attention_mask 数据。

    并且在【上篇】有所提及：
    phase3存储的数据是经过flip翻转的、是倒序的，
    后续将在data_collator中先padding后再flip回正序，
    这将使得pad_token位于数据前侧。
    """
    # 创建一个用于训练的数据集
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, # 当前进程在分布式训练环境中的排名。在非分布式设置中，通常为-1。
        args.data_path, # 存储数据文件的路径
        args.data_split, # 数据分割
        args.data_output_path, # 保存输出文件的路径
        train_phase, # 训练的阶段
        args.seed, # 随机数生成器的种子
        tokenizer, # 分词器
        args.max_prompt_seq_len) # 模型训练中序列的最大长度

    # print('prompt_train_dataset is', prompt_train_dataset)
    # prompt_train_dataset is <utils.data.data_utils.PromptDataset object at 0x7f5d3c0aa790>

    # 是否启用了无监督训练
    if unsupervised_training_enabled:
        """
        如果启用无监督训练，则获取无监督数据，
        并将其处理成分块形式，
        每块为1条数据，为max_seq_len长度
        """
        # 获取无监督训练数据
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
        print('unsupervised_train_dataset is', unsupervised_train_dataset)
    else:
        # 不使用无监督训练数据
        unsupervised_train_dataset = None

    """实例化数据整理器data_collator"""
    # DataLoaders creation:
    # 将一批数据整理成模型可以接收的形式
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    # print('data_collator is', data_collator)
    # data_collator is <utils.data.data_utils.DataCollatorRLHF object at 0x7fda5203df40>

    # 创建数据集的采样器（Sampler）
    # 情况1 : 并未使用分布式训练
    if args.local_rank == -1:
        # 对数据进行随机采样
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        # 如果启用了无监督训练，则也会为无监督训练数据集创建一个对应的采样器
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(unsupervised_train_dataset)
    # 情况2 : 使用分布式训练
    else:
        # DistributedSampler会确保在多个进程中，每个样本都只会被采样一次，这是分布式训练中避免数据冗余的重要机制。
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(unsupervised_train_dataset)

    # print('prompt_train_sampler is', prompt_train_sampler)
    #prompt_train_sampler is <torch.utils.data.distributed.DistributedSampler object at 0x7fda5203deb0>

    """
    实例化数据加载器dataloader
    并且使用data_collator整理读取到的prompt数据（如上述所说：先padding后flip）
    """
    # 在训练模型时批量加载数据
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset, # 训练数据集
        collate_fn=data_collator, # 把不同长度的句子通过padding整理成等长的句子
        sampler=prompt_train_sampler, # 从数据集中抽取数据的顺序
        batch_size=args.per_device_train_batch_size)

    # 启用了无监督训练
    if unsupervised_training_enabled:
        """如果启用无监督训练，则实例化无监督数据加载器"""
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset, # 无监督的训练数据集
            collate_fn=default_data_collator, # 默认的数据整理函数
            sampler=unsupervised_train_sampler, # 从数据集中取出数据的顺序
            batch_size=args.per_device_train_batch_size)
    else:
        """
        如果未启用无监督训练，也仍实例化一个空的数据加载器，
        因为多数后续代码入参接口都支持同时输入prompt数据与无监督数据，
        这一步是基于后续传参的安全性考虑
        """
        # 未启用无监督训练，将创建一个dummy的数据加载器
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader
    # 计算每个训练epoch需要进行的更新步数
    # args.per_device_train_batch_size/args.per_device_mini_train_batch_size：计算每个大批次需要被划分成多少个小批次
    # args.ppo_epochs：在使用PPO算法进行训练时，每个大批次中要进行的更新次数。
    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    # 总的训练迭代次数
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    # print('prompt_train_dataloader is', prompt_train_dataloader)
    # prompt_train_dataloader is <torch.utils.data.dataloader.DataLoader object at 0x7fda5203dbb0>

    # unsupervised_train_dataloader is [None, None, None, None, None, None, None, None, None, ...]
    # print('unsupervised_train_dataloader is', unsupervised_train_dataloader)
    print('len of unsupervised_train_dataloader is', len(unsupervised_train_dataloader))

    print('num_update_steps_per_epoch is', num_update_steps_per_epoch)
    print('num_total_iters is', num_total_iters)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    args = parse_args()

    # 非分布式训练
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        # 进行分布式训练
        # local_rank在这里表示当前进程在分布式训练中使用的GPU的ID
        torch.cuda.set_device(args.local_rank)

        device = torch.device("cuda", args.local_rank)

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # 初始化分布式环境
        logflag=f'ph3 deepspeed.init_distributed()'
        if args.local_rank == 0:
            gd.enable_times(info=logflag)
        deepspeed.init_distributed()
        if args.local_rank == 0:
            gd.disable_times(info=logflag)

    # 在分布式训练环境中获取当前进程的全局唯一标识符，即全局进程ID。
    # torch.distributed.get_rank()返回的是一个整数，这个整数在整个分布式训练环境中是唯一的，用于表示当前进程。
    # 对于非分布式训练环境，这个函数默认返回0。
    args.global_rank = torch.distributed.get_rank()
    gd.debuginfo(prj="ds_chat", info=f"args.global_rank is:, {args.global_rank}")

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    gd.debuginfo(prj="ds_chat", info=f"unsupervised_training_enabled is:, {unsupervised_training_enabled}")

    # 如果启用无监督训练
    if unsupervised_training_enabled:
        gd.debuginfo(prj="ds_chat")
        # if we enable unsupervised training, we need to double the batch size for actor model
        # actor模型的梯度积累步数将设置为原始梯度积累步数的两倍
        # 这是因为在无监督训练中，需要处理更多的数据（即，每个批次的数据量将加倍），因此需要更多的梯度积累步数来平衡更新。
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        gd.debuginfo(prj="ds_chat")
        # 如果未启用无监督训练，actor模型的梯度积累步数将与原始梯度积累步数相同。
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    gd.debuginfo(prj="ds_chat", info=f"args.gradient_accumulation_steps_actor is:\
                 {args.gradient_accumulation_steps_actor}")

    # If passed along, set the training seed now.
    # 设置训练的随机种子
    set_random_seed(args.seed)

    # 一个同步函数，它会阻塞调用它的每个进程，直到所有的进程都调用了此函数，
    # 这样就能保证所有的进程在进入下一步前都已经完成了当前步骤。
    # 在分布式训练中，这个函数常常被用来确保所有的进程在开始训练前都已经完成了数据的准备工作。
    torch.distributed.barrier()

    # create common tokenizer based on actor model
    # 创建了一个基于actor模型的分词器
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path,
                                  fast_tokenizer=True)
    gd.debuginfo(prj="ds_chat", info=f"ph3 tokenizer -1 is:, {tokenizer}")

    # 超出原序列长度的部分就会被填充为eos_token。
    tokenizer.pad_token = tokenizer.eos_token

    # make sure tokenizer is right pad in our logic
    # 填充方式为右填充
    tokenizer.padding_side = 'right'

    gd.debuginfo(prj="ds_chat", info=f"ph3 tokenizer -2 is:, {tokenizer}")

    # 创建数据集，并获取训练数据dataloader以及总的迭代次数
    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)


    # RLHF engine is responsible for creating models,
    # loading checkpoints, ds-initialize models/optims/lr-schedulers
    """
    使用DeepSpeedRLHFEngine类直接初始化模型
    当然其内部仍旧调用了“create_hf_model”方法来读取模型，
    但其中实现了更为精细的DeepSpeed控制
    """
    logflag = f'rlhf_engine = DeepSpeedRLHFEngine'
    if args.local_rank == 0:
        gd.enable_times(info=logflag)

    # 4.3.1初始化DeepSpeedRLHFEngine：
    # 获得一个DeepSpeedRLHFEngine对象，用于初始化一系列模型，包括Actor、Critic、Reference和Reward。
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path, # actor模型
        critic_model_name_or_path=args.critic_model_name_or_path, # critic模型
        tokenizer=tokenizer, # 分词器
        num_total_iters=num_total_iters, # 总的训练迭代次数
        args=args)
    if args.local_rank == 0:
        gd.disable_times(info=logflag)

    gd.debuginfo(prj="ds_chat", info=f"rlhf_engine is:, {rlhf_engine}")
    # rlhf_engine is: <rlhf_engine.DeepSpeedRLHFEngine object at 0x7ffaf9d97bb0>

    # 该字段的值为一个空字符串，用于表示一个对话的结束
    args.end_of_conversation_token = "<|endoftext|>"

    # 根据是否启用了无监督训练，选择了不同的PPO训练器类进行实例化。
    # ① 启用无监督 : 针对无监督训练环境（即模型只根据自身生成的数据进行训练，而不依赖人工标注的数据）设计的PPO训练器
    # ② 没有启用无监督 : 一个更通用的PPO训练器
    logflag = f'trainer = ppo_trainer(rlhf_engine, args)'
    if args.local_rank == 0:
        gd.enable_times(info=logflag)
    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)
    logflag = f'trainer.train_unsupervised'
    if args.local_rank == 0:
        gd.disable_times(info=logflag)


    gd.debuginfo(prj="ds_chat", info=f"ppo_trainer is:, {ppo_trainer}")
    gd.debuginfo(prj="ds_chat", info=f"trainer is:, {trainer}")
    # ppo_trainer is: <class 'ppo_trainer.DeepSpeedPPOTrainer'>
    # trainer is: <ppo_trainer.DeepSpeedPPOTrainer object at 0x7f939c0b7160>

    # first number is how many experience-batch to generate,
    # second number is the training batch size, which is the micro-batch size used
    #经验数据以及无监督数据都将被MiniDataset所管理
    exp_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                   args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                     args.per_device_mini_train_batch_size)

    gd.debuginfo(prj="ds_chat", info=f"exp_mini_dataset is:, {exp_mini_dataset}")
    gd.debuginfo(prj="ds_chat", info=f"unsup_mini_dataset is:, {unsup_mini_dataset}")

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    # 训练的总Epoch数
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)

        # 遍历每一个Batch
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):

            batch_prompt = to_device(batch_prompt, device)
            if batch_unsupervised is not None:
                gd.debuginfo(prj="ds_chat")
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                gd.debuginfo(prj="ds_chat")
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_train_batch_size])

            if unsup_dataset: #可能是None
                gd.debuginfo(prj="ds_chat", info=f"len of unsup_dataset: {len(unsup_dataset)}")
                #len of unsup_dataset 1
                gd.debuginfo(prj="ds_chat", info=f"unsup_dataset: {unsup_dataset}")
                # unsup_dataset [[[None, None, None, None]]]
            else:
                gd.debuginfo(prj="ds_chat", info=f"unsup_dataset is None")


            # prompts = batch_prompt['prompt']
            # length = prompts.size(-1)
            # if length > args.max_prompt_seq_len:
            #     prompts = prompts[:, length - args.max_prompt_seq_len:]
            #     raise ValueError("Prompt length is too long")
            # out为经验数据
            # 进行采样，并加入到经验池，详见（3.1）
            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'],
                                              step)
            
            gd.debuginfo(prj="ds_chat", info=f"out of generate_experience :, {out}")

            gd.debuginfo(prj="ds_chat", info=f"T out['prompts']:, {infoTensor(out['prompts'])}")
            gd.debuginfo(prj="ds_chat", info=f"T out['logprobs']:, {infoTensor(out['logprobs'])}")
            gd.debuginfo(prj="ds_chat", info=f"T out['ref_logprobs']:, {infoTensor(out['ref_logprobs'])}")
            gd.debuginfo(prj="ds_chat", info=f"T out['value']:, {infoTensor(out['value'])}")
            gd.debuginfo(prj="ds_chat", info=f"T out['rewards']:, {infoTensor(out['rewards'])}")
            gd.debuginfo(prj="ds_chat", info=f"T out['input_ids']:, {infoTensor(out['input_ids'])}")
            gd.debuginfo(prj="ds_chat", info=f"T out['attention_mask']:, {infoTensor(out['attention_mask'])}")
            ''' only ph3 x1
            T out['prompts']: _Size([4, 256])_int64_cuda:1_
            T out['logprobs']: _Size([4, 511])_float16_cuda:1_
            T out['ref_logprobs']: _Size([4, 511])_float16_cuda:1_
            T out['value']: _Size([4, 511])_float16_cuda:1_
            T out['rewards']: _Size([4])_float16_cuda:1_
            T out['input_ids']: _Size([4, 512])_int64_cuda:1_
            T out['attention_mask']: _Size([4, 512])_int64_cuda:1_
            '''

            exp_dataset = exp_mini_dataset.add(out)
            gd.debuginfo(prj="ds_chat", info=f"exp_dataset is:, {exp_dataset}")
            if exp_dataset: #可能是None
                gd.debuginfo(prj="ds_chat", info=f"len of exp_dataset: {len(exp_dataset)}")
                gd.debuginfo(prj="ds_chat", info=f"T exp_dataset[0]['prompts']:, {infoTensor(exp_dataset[0]['prompts'])}")
                gd.debuginfo(prj="ds_chat", info=f"T exp_dataset[0]['logprobs']:, {infoTensor(exp_dataset[0]['logprobs'])}")
                gd.debuginfo(prj="ds_chat", info=f"T exp_dataset[0]['ref_logprobs']:, {infoTensor(exp_dataset[0]['ref_logprobs'])}")
                gd.debuginfo(prj="ds_chat", info=f"T exp_dataset[0]['value']:, {infoTensor(exp_dataset[0]['value'])}")
                gd.debuginfo(prj="ds_chat", info=f"T exp_dataset[0]['rewards']:, {infoTensor(exp_dataset[0]['rewards'])}")
                gd.debuginfo(prj="ds_chat", info=f"T exp_dataset[0]['attention_mask']:, {infoTensor(exp_dataset[0]['attention_mask'])}")
            '''
            T exp_dataset[0]['prompts']: _Size([4, 256])_int64_cuda:0_
            T exp_dataset[0]['logprobs']: _Size([4, 511])_float16_cuda:0_
            T exp_dataset[0]['ref_logprobs']: _Size([4, 511])_float16_cuda:0_
            T exp_dataset[0]['value']: _Size([4, 511])_float16_cuda:0_
            T exp_dataset[0]['rewards']: _Size([4])_float16_cuda:0_
            T exp_dataset[0]['attention_mask']: _Size([4, 512])_int64_cuda:0_
            '''

            if exp_dataset is not None:
                gd.debuginfo(prj="ds_chat")
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0

                if args.actor_gradient_checkpointing:
                    gd.debuginfo(prj="ds_chat")
                    rlhf_engine.actor.gradient_checkpointing_enable()

                '''
                3.3.5 PPO训练过程
                
                3.3.5.1 基本流程
                
                对于采集到的一批经验数据，使用MiniDataset处理成多批ppo_batch数据，供相关模型进行多次训练迭代，更具体的训练细节见后续内容。
                
                而DeepSpeed-Chat中所设置的ppo_epochs，从强化学习的角度来说，实际上代表的是一批经验数据的复用次数：
                
                    假如ppo_epochs设置为1，训练时，引入的这批经验数据在经过1次全遍历后，
                    将被直接弃置，随之进行下一轮prompt_epoch，届时将重新采集新的一批经验数据；
                    
                    假如ppo_epochs设置为n，训练时，引入的这批经验数据将被遍历n次才被弃置，
                    即相当于这批经验数据被复用了n次用于off-policy训练。
                    
                '''
                # 从经验池中进行学习Epoch轮
                for ppo_ep in range(args.ppo_epochs):
                    gd.debuginfo(prj="ds_chat", info = f"ppo_ep is {ppo_ep}")
                    #ppo_epoch循环
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        gd.debuginfo(prj="ds_chat")
                        gd.debuginfo(prj="ds_chat", info=f"exp_data is:, {exp_data}")
                        gd.debuginfo(prj="ds_chat", info=f"unsup_dataset is:, {unsup_dataset}")

                        gd.debuginfo(prj="ds_chat", info=f"T exp_data['prompts']:, {infoTensor(exp_data['prompts'])}")
                        gd.debuginfo(prj="ds_chat", info=f"T exp_data['logprobs']:, {infoTensor(exp_data['logprobs'])}")
                        gd.debuginfo(prj="ds_chat", info=f"T exp_data['ref_logprobs']:, {infoTensor(exp_data['ref_logprobs'])}")
                        gd.debuginfo(prj="ds_chat", info=f"T exp_data['value']:, {infoTensor(exp_data['value'])}")
                        gd.debuginfo(prj="ds_chat", info=f"T exp_data['rewards']:, {infoTensor(exp_data['rewards'])}")
                        gd.debuginfo(prj="ds_chat", info=f"T exp_data['input_ids']:, {infoTensor(exp_data['input_ids'])}")
                        gd.debuginfo(prj="ds_chat", info=f"T exp_data['attention_mask']:, {infoTensor(exp_data['attention_mask'])}")
                        '''
                        T exp_data['prompts']: _Size([4, 256])_int64_cuda:1_
                        T exp_data['logprobs']: _Size([4, 511])_float16_cuda:1_
                        T exp_data['ref_logprobs']: _Size([4, 511])_float16_cuda:1_
                        T exp_data['value']: _Size([4, 511])_float16_cuda:1_
                        T exp_data['rewards']: _Size([4])_float16_cuda:1_
                        T exp_data['input_ids']: _Size([4, 512])_int64_cuda:1_
                        T exp_data['attention_mask']: _Size([4, 512])_int64_cuda:1_
                        '''

                        """
                        ppo_step循环：
                        从MiniDataset返回的数据中，
                        取1个ppo_batch的经验数据和无监督数据来训练。
                        """

                        #经验数据训练，返回actor_loss和critic_loss
                        # 得到actor和critic loss，详见（3.2）
                        logflag = f'trainer.train_rlhf'
                        if args.local_rank == 0:
                            gd.enable_times(info=logflag)
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        if args.local_rank == 0:
                            gd.disable_times(info=logflag)

                        #累加本ppo_step的指标，后续将除以内层迭代次数计算均值
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()

                        #无监督数据训练
                        if unsupervised_training_enabled:
                            # 返回无监督损失
                            logflag = f'trainer.train_unsupervised'
                            if args.local_rank == 0:
                                gd.enable_times(info=logflag)
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            if args.local_rank == 0:
                                gd.disable_times(info=logflag)


                            gd.debuginfo(prj="ds_chat", info=f"unsup_loss is {unsup_loss}")

                            #累加本ppo_step的无监督损失，后续将除以内层迭代次数计算均值
                            unsup_loss_sum += unsup_loss.item()

                        # PPO训练迭代次数（ppo_step）+1
                        inner_iter += 1

                        """是否启用指数移动平均技术"""
                        if args.enable_ema:
                            gd.debuginfo(prj="ds_chat", info=f"enable_ema")
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    # 打乱数据供off - policy复用
                    # 每一轮结束后打乱经验池
                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                print_rank_0(
                    f'epoch: {epoch}|step: {step}|ppo_ep: {ppo_ep+1}|act_loss: {actor_loss_sum/inner_iter}|cri_loss: {critic_loss_sum/inner_iter}|unsuper_loss: {unsup_loss_sum/inner_iter}',
                    args.global_rank)

                average_reward = get_all_reduce_mean(average_reward).item()

                print_rank_0(f"average reward score: {average_reward/inner_iter}",args.global_rank)
                print_rank_0("--------------------------------------------------------",args.global_rank)

                if args.enable_tensorboard and torch.distributed.get_rank() == 0:
                    gd.debuginfo(prj="ds_chat")
                    writer.add_scalar('reward',
                                      average_reward / inner_iter,
                                      global_step=step)
                    writer.add_scalar('actor_loss',
                                      actor_loss,
                                      global_step=step)
                    writer.add_scalar('actor_loss_sum',
                                      actor_loss_sum,
                                      global_step=step)
                    writer.add_scalar('critic_loss',
                                      critic_loss,
                                      global_step=step)
                    writer.add_scalar('critic_loss_sum',
                                      critic_loss_sum,
                                      global_step=step)
                    writer.flush()

            if args.actor_gradient_checkpointing:
                gd.debuginfo(prj="ds_chat")
                rlhf_engine.actor.gradient_checkpointing_disable()

    if args.output_dir is not None:
        print_rank_0('saving model ...')

        gd.debuginfo(prj="ds_chat", info=f"ph3 rlhf_engine.actor model:, {rlhf_engine.actor}")
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        gd.debuginfo(prj="ds_chat", info=f"ph3 actor convert_lora_to_linear_layer model:, {rlhf_engine.actor}")

        gd.debuginfo(prj="ds_chat", info=f"ph3 rlhf_engine.critic model:, {rlhf_engine.critic}")
        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
        gd.debuginfo(prj="ds_chat", info=f"ph3 critic convert_lora_to_linear_layer model:, {rlhf_engine.critic}")

        if args.enable_ema:
            gd.debuginfo(prj="ds_chat")
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            gd.debuginfo(prj="ds_chat")
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')

            save_hf_format(rlhf_engine.critic,
                           tokenizer,
                           args,
                           sub_folder='critic')

            if args.enable_ema:
                gd.debuginfo(prj="ds_chat")
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')

        if args.actor_zero_stage == 3:
            gd.debuginfo(prj="ds_chat")
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                gd.debuginfo(prj="ds_chat")
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=args.global_rank,
                                      save_dir=os.path.join(
                                          args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)

        if args.critic_zero_stage == 3:
            gd.debuginfo(prj="ds_chat")
            save_zero_three_model(rlhf_engine.critic,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)

    gd.dumpCounter()


if __name__ == "__main__":
    main()
	
	
'''
exp_dataset is: [
{'prompts': tensor([[    2,     2,     2,  ..., 50118, 46184,    35],
        ...
        [    2,     2,     2,  ..., 50118, 46184,    35]], device='cuda:0'), 
'logprobs': tensor([[-5.9297e+00, -5.9297e+00, -5.9297e+00,  ..., -5.8479e-03,
         -1.4099e-02, -3.5980e-02],
        ...
        [-4.4873e-01, -4.4873e-01, -4.4873e-01,  ..., -1.6678e-02,
         -2.6123e-02, -1.9791e-02]], device='cuda:0', dtype=torch.float16),
'ref_logprobs': tensor([[-5.9297e+00, -5.9297e+00, -5.9297e+00,  ..., -5.8479e-03,
        ...
        [-4.4873e-01, -4.4873e-01, -4.4873e-01,  ..., -1.6678e-02,
         -2.6123e-02, -1.9791e-02]], device='cuda:0', dtype=torch.float16), 
'value': tensor([[-0.1948, -0.1948, -0.1948,  ...,  0.8779,  0.9243,  0.9229],
        ...
        [ 0.4233,  0.4233,  0.4233,  ...,  0.5342,  0.6001,  0.5879]],
       device='cuda:0', dtype=torch.float16),
'rewards': tensor([0.7358, 0.8081, 0.0402, 0.4734], device='cuda:0', dtype=torch.float16), 'input_ids': tensor([[    2,     2,     2,  ...,    64,    67, 10397],
        ...
        [    2,     2,     2,  ...,    10,   357,  4885]], device='cuda:0'),
'attention_mask': tensor([[0, 0, 0,  ..., 1, 1, 1],
        ...
        [0, 0, 0,  ..., 1, 1, 1]], device='cuda:0')}]
'''

'''
exp_data is: {
'prompts': tensor([[    2,     2,     2,  ..., 50118, 46184,    35],
        ...
        [    2,     2,     2,  ..., 50118, 46184,    35]], device='cuda:0'), 
'logprobs': tensor([[-5.1875e+00, -5.1875e+00, -5.1875e+00,  ..., -7.6709e-01,
         -7.7271e-02, -1.6201e+00],
        ...
        [-1.1699e+00, -1.1699e+00, -1.1699e+00,  ..., -7.5006e-04,
         -7.5684e-02, -3.6438e-02]], device='cuda:0', dtype=torch.float16), 
'ref_logprobs': tensor([[-5.1641e+00, -5.1641e+00, -5.1641e+00,  ..., -8.6523e-01,
         -1.0699e-01, -1.7266e+00],
        ...
        [-1.1084e+00, -1.1084e+00, -1.1084e+00,  ..., -7.0190e-04,
         -1.5137e-01, -6.1401e-02]], device='cuda:0', dtype=torch.float16), 
'value': tensor([[-0.2289, -0.2289, -0.2289,  ...,  0.2812,  0.3298,  0.4016],
        ...
        [ 1.1162,  1.1162,  1.1162,  ...,  0.7139,  0.6431,  0.6313]],
       device='cuda:0', dtype=torch.float16), 
'rewards': tensor([0.3215, 0.9287, 1.0088, 0.5864], device='cuda:0', dtype=torch.float16), 
'input_ids': tensor([[    2,     2,     2,  ...,     9,     5,   144],
        ...
        [    2,     2,     2,  ...,    35,   318,    47]], device='cuda:0'), 
'attention_mask': tensor([[0, 0, 0,  ..., 1, 1, 1],
        ...
        [0, 0, 0,  ..., 1, 1, 1]], device='cuda:0')}
        
unsup_dataset is: [[[None, None, None, None]]]
'''
	
	
	
'''
out of generate_experience : {
'prompts': tensor([[    2,     2,     2,  ..., 50118, 46184,    35],
        ...
        [    2,     2,     2,  ..., 50118, 46184,    35]], device='cuda:1'), 
'logprobs': tensor([[-5.8633e+00, -5.8633e+00, -5.8633e+00,  ..., -1.2253e-02,
         -1.6815e-02, -3.3474e-03],
        ...
        [-2.0078e+00, -2.0078e+00, -2.0078e+00,  ..., -1.9516e-02,
         -2.6443e-02, -6.1607e-03]], device='cuda:1', dtype=torch.float16), 
'ref_logprobs': tensor([[-5.8633e+00, -5.8633e+00, -5.8633e+00,  ..., -1.3519e-02,
         -1.8341e-02, -3.7823e-03],
        ...
        [-2.0391e+00, -2.0391e+00, -2.0391e+00,  ..., -2.1149e-02,
         -2.8412e-02, -6.6376e-03]], device='cuda:1', dtype=torch.float16), 
'value': tensor([[-0.4321, -0.4321, -0.4321,  ..., -0.4736, -0.4829, -0.3918],
        ...
        [ 1.5361,  1.5361,  1.5361,  ...,  1.2910,  1.3447,  1.2559]],device='cuda:1', dtype=torch.float16), 
'rewards': tensor([-0.4282,  1.4141, -0.3965,  1.2178], device='cuda:1',dtype=torch.float16), 
'input_ids': tensor([[   2,    2,    2,  ...,   47,   64,   67],
        ...
        [   2,    2,    2,  ...,    7, 7142,   24]], device='cuda:1'), 
'attention_mask': tensor([[0, 0, 0,  ..., 1, 1, 1],
        ...
        [0, 0, 0,  ..., 1, 1, 1]], device='cuda:1')}
'''
