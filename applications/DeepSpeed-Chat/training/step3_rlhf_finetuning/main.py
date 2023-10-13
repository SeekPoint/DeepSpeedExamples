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

from pydebug import debuginfo, infoTensor

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

    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data. For example the split `2,4,4` '
        'will use 60% of data for phase 1, 20% for phase 2 and 20% for phase 3.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. We did not see this in other models but keep it as an option for now."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_mini_train_batch_size",
        type=int,
        default=16,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batch_numbers",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    parser.add_argument('--disable_actor_dropout',
                        action='store_true',
                        help='Disable the dropout of the actor model.')
    parser.add_argument('--disable_critic_dropout',
                        action='store_true',
                        help='Disable the dropout of the critical model.')
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    ## Make EMA as an optional feature
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
    args = parser.parse_args()

    if args.enable_tensorboard:
        print(
            f"Tensorboard logs going to: {args.tensorboard_path}/step3_tensorboard_logs"
        )
        writer = SummaryWriter(
            f"{args.tensorboard_path}/step3_tensorboard_logs")

    # Validate settings
    if (args.actor_gradient_checkpointing
            and args.actor_lora_dim > 0) or (args.critic_gradient_checkpointing
                                             and args.critic_lora_dim > 0):
        assert (
            not args.only_optimize_lora
        ), "--{actor,critic}_gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    return args

# 3.3.1.3 无监督数据集的读取
# 无监督数据集主要是进行了分块处理，将无监督语料全部拼接起来得到一个极长的序列，
# 使用max_seq_len大小的滑窗对长序列进行分块，每个分块将作为1条无监督数据。
def create_datasets(args, tokenizer, train_phase=3):
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

    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len)
    # print('prompt_train_dataset is', prompt_train_dataset)
    # prompt_train_dataset is <utils.data.data_utils.PromptDataset object at 0x7f5d3c0aa790>

    if unsupervised_training_enabled:
        """
        如果启用无监督训练，则获取无监督数据，
        并将其处理成分块形式，
        每块为1条数据，为max_seq_len长度
        """
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
        print('unsupervised_train_dataset is', unsupervised_train_dataset)
    else:
        unsupervised_train_dataset = None

    """实例化数据整理器data_collator"""
    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    # print('data_collator is', data_collator)
    # data_collator is <utils.data.data_utils.DataCollatorRLHF object at 0x7fda5203df40>

    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)

    #print('prompt_train_sampler is', prompt_train_sampler)
    #prompt_train_sampler is <torch.utils.data.distributed.DistributedSampler object at 0x7fda5203deb0>

    """
    实例化数据加载器dataloader
    并且使用data_collator整理读取到的prompt数据（如上述所说：先padding后flip）
    """
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_train_batch_size)

    if unsupervised_training_enabled:
        """如果启用无监督训练，则实例化无监督数据加载器"""
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_train_batch_size)
    else:
        """
        如果未启用无监督训练，也仍实例化一个空的数据加载器，
        因为多数后续代码入参接口都支持同时输入prompt数据与无监督数据，
        这一步是基于后续传参的安全性考虑
        """
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    # print('prompt_train_dataloader is', prompt_train_dataloader)
    # prompt_train_dataloader is <torch.utils.data.dataloader.DataLoader object at 0x7fda5203dbb0>


    # unsupervised_train_dataloader is [None, None, None, None, None, None, None, None, None, ...]
    #print('unsupervised_train_dataloader is', unsupervised_train_dataloader)
    print('len of unsupervised_train_dataloader is', len(unsupervised_train_dataloader))

    print('num_update_steps_per_epoch is', num_update_steps_per_epoch)
    print('num_total_iters is', num_total_iters)


    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # create common tokenizer based on actor model
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path,
                                  fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'
    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    """
    使用DeepSpeedRLHFEngine类直接初始化模型
    当然其内部仍旧调用了“create_hf_model”方法来读取模型，
    但其中实现了更为精细的DeepSpeed控制
    """

    # 4.3.1初始化DeepSpeedRLHFEngine：
    # 获得一个DeepSpeedRLHFEngine对象，用于初始化一系列模型，包括Actor、Critic、Reference和Reward。
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    # print("rlhf_engine is:", rlhf_engine)
    # rlhf_engine is: <rlhf_engine.DeepSpeedRLHFEngine object at 0x7ffaf9d97bb0>

    args.end_of_conversation_token = "<|endoftext|>"

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # print("ppo_trainer is:", ppo_trainer)
    # print("trainer is:", trainer)
    # ppo_trainer is: <class 'ppo_trainer.DeepSpeedPPOTrainer'>
    # trainer is: <ppo_trainer.DeepSpeedPPOTrainer object at 0x7f939c0b7160>

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    #经验数据以及无监督数据都将被MiniDataset所管理
    exp_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                   args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                     args.per_device_mini_train_batch_size)

    print("exp_mini_dataset is:", exp_mini_dataset)
    print("unsup_mini_dataset is:", unsup_mini_dataset)

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
                debuginfo(prj='ds-chat')
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                debuginfo(prj='ds-chat')
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_train_batch_size])
            # print("len of unsup_dataset", len(unsup_dataset))
            #len of unsup_dataset 1
            #print("unsup_dataset", unsup_dataset)
            # unsup_dataset [[[None, None, None, None]]]


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
            # print("out of generate_experience :", out)
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


            # print("T out['prompts']:", infoTensor(out['prompts']))
            # print("T out['logprobs']:", infoTensor(out['logprobs']))
            # print("T out['ref_logprobs']:", infoTensor(out['ref_logprobs']))
            # print("T out['value']:", infoTensor(out['value']))
            # print("T out['rewards']:", infoTensor(out['rewards']))
            # print("T out['input_ids']:", infoTensor(out['input_ids']))
            # print("T out['attention_mask']:", infoTensor(out['attention_mask']))
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
            #print("exp_dataset is:", exp_dataset)
            # print("len of exp_dataset", len(exp_dataset))
            # len of exp_dataset
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

            # print("T exp_dataset[0]['prompts']:", infoTensor(exp_dataset[0]['prompts']))
            # print("T exp_dataset[0]['logprobs']:", infoTensor(exp_dataset[0]['logprobs']))
            # print("T exp_dataset[0]['ref_logprobs']:", infoTensor(exp_dataset[0]['ref_logprobs']))
            # print("T exp_dataset[0]['value']:", infoTensor(exp_dataset[0]['value']))
            # print("T exp_dataset[0]['rewards']:", infoTensor(exp_dataset[0]['rewards']))
            # print("T exp_dataset[0]['attention_mask']:", infoTensor(exp_dataset[0]['attention_mask']))
            '''
            T exp_dataset[0]['prompts']: _Size([4, 256])_int64_cuda:0_
            T exp_dataset[0]['logprobs']: _Size([4, 511])_float16_cuda:0_
            T exp_dataset[0]['ref_logprobs']: _Size([4, 511])_float16_cuda:0_
            T exp_dataset[0]['value']: _Size([4, 511])_float16_cuda:0_
            T exp_dataset[0]['rewards']: _Size([4])_float16_cuda:0_
            T exp_dataset[0]['attention_mask']: _Size([4, 512])_int64_cuda:0_
            '''

            if exp_dataset is not None:
                debuginfo(prj='ds-chat')
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0

                if args.actor_gradient_checkpointing:
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
                    debuginfo(prj='ds-chat', info = f"ppo_ep is {ppo_ep}")
                    #ppo_epoch循环
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        debuginfo(prj='ds-chat')
                        # print("exp_data is:", exp_data)
                        # print("unsup_dataset is:", unsup_dataset)
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

                        # print("T exp_data['prompts']:", infoTensor(exp_data['prompts']))
                        # print("T exp_data['logprobs']:", infoTensor(exp_data['logprobs']))
                        # print("T exp_data['ref_logprobs']:", infoTensor(exp_data['ref_logprobs']))
                        # print("T exp_data['value']:", infoTensor(exp_data['value']))
                        # print("T exp_data['rewards']:", infoTensor(exp_data['rewards']))
                        # print("T exp_data['input_ids']:", infoTensor(exp_data['input_ids']))
                        # print("T exp_data['attention_mask']:", infoTensor(exp_data['attention_mask']))
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
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)

                        #累加本ppo_step的指标，后续将除以内层迭代次数计算均值
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()

                        #无监督数据训练
                        if unsupervised_training_enabled:
                            # 返回无监督损失
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)

                            debuginfo(prj='ds-chat', info=f"unsup_loss is {unsup_loss}")

                            #累加本ppo_step的无监督损失，后续将除以内层迭代次数计算均值
                            unsup_loss_sum += unsup_loss.item()

                        # PPO训练迭代次数（ppo_step）+1
                        inner_iter += 1

                        """是否启用指数移动平均技术"""
                        if args.enable_ema:
                            debuginfo(prj='ds-chat', info=f"enable_ema")
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
                print_rank_0(
                    f"average reward score: {average_reward/inner_iter}",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)
                if args.enable_tensorboard and torch.distributed.get_rank(
                ) == 0:
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
                rlhf_engine.actor.gradient_checkpointing_disable()

    if args.output_dir is not None:
        print_rank_0('saving model ...')
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)
        rlhf_engine.critic = convert_lora_to_linear_layer(rlhf_engine.critic)
        if args.enable_ema:
            rlhf_engine.actor_ema = convert_lora_to_linear_layer(
                rlhf_engine.actor_ema)

        if torch.distributed.get_rank() == 0:
            debuginfo(prj='ds-chat')
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')
            save_hf_format(rlhf_engine.critic,
                           tokenizer,
                           args,
                           sub_folder='critic')
            if args.enable_ema:
                save_hf_format(rlhf_engine.actor_ema,
                               tokenizer,
                               args,
                               sub_folder='actor_ema')

        if args.actor_zero_stage == 3:
            debuginfo(prj='ds-chat')
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)
            if args.enable_ema:
                save_zero_three_model(rlhf_engine.actor_ema,
                                      global_rank=args.global_rank,
                                      save_dir=os.path.join(
                                          args.output_dir, 'actor_ema'),
                                      zero_stage=args.actor_zero_stage)

        if args.critic_zero_stage == 3:
            debuginfo(prj='ds-chat')
            save_zero_three_model(rlhf_engine.critic,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'critic'),
                                  zero_stage=args.critic_zero_stage)


if __name__ == "__main__":
    main()
