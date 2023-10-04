#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# 首先，它导入了Python的标准库，如argparse（用于解析命令行参数），os和math。
# 然后，它导入了PyTorch库，这是一个用于深度学习的开源库，同时也导入了一些PyTorch的辅助模块，如DataLoader（用于加载数据）、
# RandomSampler和SequentialSampler（用于数据抽样）以及DistributedSampler（用于在分布式设置中进行数据抽样）。
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

# 接下来，它导入了Hugging Face的transformers库的一些模块，包括用于因果语言建模的模型（AutoModelForCausalLM），优化器调度类型（SchedulerType），默认的数据整理函数（default_data_collator）和获取优化器调度器的函数（get_scheduler）。
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

# 然后，它导入了deepspeed库，这是一个为大规模模型训练优化的库。它也导入了deepspeed库中的一些模块，包括优化器类（DeepSpeedCPUAdam和FusedAdam）
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# 之后，它将当前脚本的父目录添加到系统路径中，以便可以从该目录下的utils目录导入一些自定义函数和模块。
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# 最后，它从utils目录中导入了一些自定义模块和函数，包括数据处理函数（create_prompt_dataset），打印和设备转换函数（print_rank_0和to_device），
# 模型保存函数（save_hf_format），随机种子设置函数（set_random_seed），求平均函数（get_all_reduce_mean），
# 获取优化器参数组的函数（get_optimizer_grouped_parameters），保存和加载模型的函数（save_zero_three_model和load_hf_tokenizer），
# 以及创建模型和处理模型的函数（create_hf_model）。这些函数在脚本中的后续部分都将被使用。
print("1===", os.path.pardir)
print("2===", os.path.dirname(__file__))
print("3===", os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model


def parse_args():
    # 创建一个argparse的解析器对象，这个对象可以添加命令行参数和处理它们。description参数提供了一个对程序的简单描述。
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                             '1) a single data path, 2) multiple datasets in the'
                             'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                             'phase 1, 2, and 3 data. For example the split `6,2,2`'
                             'will use 60% of data for phase 1, 20% for phase 2'
                             'and 20% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='./data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
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
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=1,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')

    from calltrace import CallTrace
    ct_ds_addconfig = CallTrace(tag='ds_addconfig')
    ct_ds_addconfig.startrecord()

    # 这一行将DeepSpeed的配置参数添加到解析器中。
    parser = deepspeed.add_config_arguments(parser)
    ct_ds_addconfig.endRecord()

    # 这一行解析命令行参数并将它们存储在args对象中
    args = parser.parse_args()

    # Validate settings
    # 在这个代码块中，验证一些特定的参数设置是否合法。
    # 例如，如果同时启用了gradient_checkpointing和仅优化LoRA参数，那么将会抛出一个错误。
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args

# 这个函数是主函数，是训练语言模型的主流程，主要步骤包括解析命令行参数、
# 设置设备、准备数据、定义模型、配置优化器和学习率调度器、进行训练和评估等。
def main():
    # 解析命令行参数。
    args = parse_args()

    # 如果本地排名为-1，说明不在分布式训练环境下，设备设置为"cuda"；
    # 否则根据args.local_rank设置对应的cuda设备，并初始化分布式训练。
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    # 获取全局rank。
    args.global_rank = torch.distributed.get_rank()
    print("args.global_rank is:", args.global_rank)

    # 获取deepspeed的训练配置。
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    # 在配置中设置训练时每个GPU的微批次大小和总的批次大小。
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    print("ds_config['train_batch_size']is:", ds_config['train_batch_size'])
    print("args.per_device_train_batch_size is:", args.per_device_train_batch_size)
    # If passed along, set the training seed now.
    # 设置随机种子以保证结果的可复现性。
    set_random_seed(args.seed)

    # 在此处添加一个barrier操作，确保所有进程都执行到这一点后再继续执行后续操作。
    torch.distributed.barrier()
    # 加载预训练模型对应的分词器。
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    print("tokenizer :", tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'
    # 创建预训练模型。
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            disable_dropout=args.disable_dropout)

    print("model---1 :", model)
    # 判断是否启用LoRA模式
    # 如果参数lora_dim大于0，将模型的线性层转换为LoRa层；如果只优化LoRa参数，关闭其他参数的梯度。
    if args.lora_dim > 0:
        '''
        yk--此处代码不同
        如果启用，则对名称中含有“decoder.layers.”且为线性层的结构部分引入LoRA旁路（实现先降维后升维的2个线性层），
        这类结构基本都是attention、信息交互用的inner线性层，
        这类结构的Weight参数将被冻结，转而优化LoRA旁路的参数。
        '''
        model = convert_linear_layer_to_lora(model, args.lora_module_name,
                                             args.lora_dim)
        if args.only_optimize_lora:
            model = only_optimize_lora_parameters(model)

    # Prepare the data
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

    print("train_dataset :", train_dataset)
    print("eval_dataset :", eval_dataset)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    print("train_sampler :", train_sampler)
    print("eval_sampler :", eval_sampler)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    print("train_dataloader :", train_dataloader)
    print("eval_dataloader :", eval_dataloader)

    '''
    1.3.3 phase1的指标评估
    DeepSpeed-Chat选择了困惑度perplexity作为phase1训练期间的评估指标。
    需要注意的是，perplexity不是绝对的评估准则，
    甚至有可能perplexity评估结果与实际情况并不一致（即，perplexity已经处于较低水平，但模型的实际生成能力却仍然堪忧），
    这点DeepSpeed-Chat团队也有做出过说明。
    
    Supervised fine-tuning (SFT) has indeed made significant progress in the field of large language models (LLMs). 
    However, unexpected behaviors such as repeating content generation and inconsistency between perplexity (PPL) scores 
    and generation capabilities can still occur.
    
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
    # 定义模型评估函数，用于计算模型在验证集上的困惑度。
    def evaluation(model, eval_dataloader):
        """
        以困惑度perplexity为评估指标进行验证
        """
        model.eval()  # 将模型切换为评估模式。
        losses = 0  # 初始化loss。
        for step, batch in enumerate(eval_dataloader):  # 对于评估数据集中的每一个batch。
            """
            batch: 由input_ids、attention_mask、labels共3个部分组成的dict。
                   其中每个部分的shape均为(bs, max_seq_len)
            """
            batch = to_device(batch, device)  # 将batch数据移到对应的设备上。
            with torch.no_grad():  # 在此上下文管理器中，不计算梯度，这样可以节省存储和计算资源。
                # 将batch数据输入模型，进行前向计算。
                outputs = model(**batch)

            """Causal LM 的损失函数为交叉熵损失"""
            loss = outputs.loss  # 取出模型的输出中的loss。
            losses += loss.float()  # 将当前的loss累加到总的losses中。
        losses = losses / (step + 1)  # 计算平均的loss。
        try:
            """困惑度perplexity通常可以通过exp(CELoss)计算得到"""
            perplexity = torch.exp(losses)  # 尝试计算模型的困惑度，如果捕捉到溢出错误，将困惑度设置为无穷大。
        except OverflowError:
            perplexity = float("inf")
        # 尝试在所有设备上计算困惑度的平均值，如果发生任何错误，就忽略。
        try:
            """
            - get_all_reduce_mean中调用了torch.distributed.all_reduce(perplexity, op=torch.distributed.ReduceOp.SUM)
            - 对所有进程、或者说GPU（因为通常情况下就是单个进程控制单个GPU）中的perplexity进行求和
            - 然后再除以全局进程数torch.distributed.get_world_size()得到平均的perplexity结果
            """
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    # 根据是否使用权重衰减将模型参数分为两组。
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)
    print("optimizer_grouped_parameters :", optimizer_grouped_parameters)
    # 选择优化器类型，如果启用了梯度Offload，使用DeepSpeedCPUAdam，否则使用FusedAdam。
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    print("AdamOptimizer :", AdamOptimizer)

    from calltrace import CallTrace
    ct_ds_AdamOptimizer = CallTrace(tag='ds_AdamOptimizer')
    ct_ds_AdamOptimizer.startrecord()
    # 创建优化器。
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    ct_ds_AdamOptimizer.endRecord()

    print("optimizer :", optimizer)
    # 计算每个epoch的更新步数。
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    print("num_update_steps_per_epoch :", num_update_steps_per_epoch)
    # 创建学习率调度器。
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    print("lr_scheduler :", lr_scheduler)

    from calltrace import CallTrace

    ct_ds_init = CallTrace(tag='ds_init')
    ct_ds_init.startrecord()
    # 使用deepspeed初始化模型、优化器和学习率调度器。
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    ct_ds_init.endRecord()

    print("model---3 :", model)
    print("optimizer---3 :", optimizer)
    print("lr_scheduler---3 :", lr_scheduler)

    # 如果启用了梯度检查点，那么在模型中也启用梯度检查点。
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    # 使用 print_rank_0 函数在主节点（global_rank为0的节点）打印开始训练的信息。
    print_rank_0("***** Running training *****", args.global_rank)
    # 在主节点打印在第0个epoch（训练开始前）进行模型评估的信息。
    print_rank_0(
        f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    # 调用 evaluation 函数对模型进行评估，得到模型的困惑度。
    perplexity = evaluation(model, eval_dataloader)
    # 在主节点打印模型的困惑度。
    print_rank_0(f"ppl: {perplexity}", args.global_rank)

    # 循环 args.num_train_epochs 轮进行训练。
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)  # 在每个epoch开始时，在主节点打印开始新的训练周期的信息。
        model.train()  # 将模型设置为训练模式。
        for step, batch in enumerate(train_dataloader):  # 对于训练数据集中的每一个batch。
            batch = to_device(batch, device)  # 将batch数据移到对应的设备上。
            outputs = model(**batch, use_cache=False)  # 将batch数据输入模型，进行前向计算。
            loss = outputs.loss  # 取出模型的输出中的loss。
            if args.print_loss:
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}"
                )
            model.backward(loss)  # 进行反向传播，计算梯度。
            model.step()  # 更新模型的参数。

        # Evaluate perplexity on the validation set.
        # 在每个epoch结束后，在主节点打印开始评估的信息。
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch + 1}/{args.num_train_epochs} *****",
            args.global_rank)
        # 对模型进行评估，得到模型的困惑度。
        perplexity = evaluation(model, eval_dataloader)
        # 在主节点打印模型的困惑度。
        print_rank_0(f"ppl: {perplexity}", args.global_rank)
        # 更新模型中的epoch计数。
        model.tput_timer.update_epoch_count()

    # 如果设置了输出目录，进行以下操作。
    if args.output_dir is not None:
        # 在主节点打印开始保存模型的信息。
        print_rank_0('saving the final model ...', args.global_rank)
        # 将模型中的LoRA层转换为线性层。
        model = convert_lora_to_linear_layer(model)
        print("model---4 :", model)
        # 如果是主节点，进行以下操作。
        if args.global_rank == 0:
            # 以Hugging Face的模型格式保存模型。
            save_hf_format(model, tokenizer, args)
        # 如果使用了Zero Redundancy Optimizer（Zero）的第三阶段，进行以下操作。
        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            # 使用特殊的保存函数保存模型。在Zero的第三阶段，每个GPU只有模型的一部分，所以需要特殊的保存函数。
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)

if __name__ == "__main__":
    main()
