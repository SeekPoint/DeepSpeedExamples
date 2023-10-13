#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from pydebug import debuginfo, infoTensor

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters


def parse_args():
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
                        'phase 1, 2, and 3 data. For example the split `2,4,4`'
                        'will use 60% of data for phase 1, 20% for phase 2'
                        'and 20% for phase 3.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help='Where to store the data-related files such as shuffle index.')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
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
        default=5e-5,
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
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
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
                        default=0,
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
                        default="step2_tensorboard")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    print("args.global_rank :", args.global_rank)

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step2_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size

    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    #print("ds_config--2 :", ds_config)
    '''
    ds_config--2 : {'train_batch_size': 16, 'train_micro_batch_size_per_gpu': 8, 'steps_per_print': 10, 'zero_optimization': {'stage': 0, 'offload_param': {'device': 'none'}, 'offload_optimizer': {'device': 'none'}, 'stage3_param_persistence_threshold': 10000.0, 'stage3_max_live_parameters': 30000000.0, 'stage3_prefetch_bucket_size': 30000000.0, 'memory_efficient_linear': False}, 'fp16': {'enabled': True, 'loss_scale_window': 100}, 'gradient_clipping': 1.0, 'prescale_gradients': False, 'wall_clock_breakdown': False, 'hybrid_engine': {'enabled': False, 'max_out_tokens': 512, 'inference_tp_size': 1, 'release_inference_cache': False, 'pin_parameters': True, 'tp_gather_partition_size': 8}, 'tensorboard': {'enabled': False, 'output_path': 'step2_tensorboard/ds_tensorboard_logs/', 'job_name': 'step2_model_tensorboard'}}
    '''

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    print("tokenizer---1 :", tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    # make sure tokenizer is right pad in our logic
    tokenizer.padding_side = 'right'
    """
    rm_model调用了create_critic_model进行载入
    默认情况下rm_model是不启用dropout的
    """
    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   disable_dropout=args.disable_dropout)

    # print("rm_model---1 :", rm_model)
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

    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        print("rm_model---2 :", rm_model)

        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
            print("rm_model---3 :", rm_model)

    train_phase = 2
    train_dataset, eval_dataset = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_seq_len)
    # print("train_dataset :", train_dataset)
    # print("eval_dataset :", eval_dataset)
    print("len of train_dataset :", len(train_dataset))
    print("len of eval_dataset :", len(eval_dataset))

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
    data_collator = DataCollatorReward()
    print("data_collator :", data_collator)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)

    # print("train_sampler :", train_sampler)
    # print("eval_sampler :", eval_sampler)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)
    # print("train_dataloader :", train_dataloader)
    # print("eval_sampler :", eval_sampler)
    # print("eval_dataloader :", eval_dataloader)
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
    def evaluation_reward(model, eval_dataloader):
        debuginfo(prj='ds-chat')
        model.eval()
        # 统计预测（赋分）正确的结果即chosen_reward > rejected_reward的结果数
        correct_predictions = 0

        # 统计预测总数
        total_predictions = 0
        scores = 0
        debuginfo(prj='ds-chat', info = len(eval_dataloader))
        for step, batch in enumerate(eval_dataloader):
            # print("batch---C is:", batch)
            #print("T batch['input_ids']--C:", infoTensor(batch['input_ids'])) #only ph2 T batch['input_ids']--C: _Size([16, 128])_int64_cpu_
            #print("T batch['attention_mask']--C:", infoTensor(batch['attention_mask'])) #pnly ph2 T batch['attention_mask']--C: _Size([16, 128])_int64_cpu_
            '''
            batch---C is: 
            {'input_ids': tensor([[    2, 50118, 50118,  ...,   533,     7, 28616],
                ...,
                [    2, 50118, 50118,  ...,  1328,     5,   183]]), 
            'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
            ...
                [1, 1, 1,  ..., 1, 1, 1]])}
            '''
            batch = to_device(batch, device)

            with torch.no_grad():
                """
                outputs: {'loss':tensor(), 
                            'chosen_mean_scores':tensor(bs,), 
                            'rejected_mean_scores':tensor(bs,)}
                """
                outputs = model(**batch)
                # print("outputs--C", outputs)
                # print("T outputs['loss']--A:", infoTensor(outputs['loss']))
                # print("T outputs['chosen_mean_scores']--A:", infoTensor(outputs['chosen_mean_scores']))
                # print("T outputs['rejected_mean_scores']--A:", infoTensor(outputs['rejected_mean_scores']))
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

            # chosen.shape: (bs,)
            chosen = outputs["chosen_mean_scores"]

            #rejected.shape: (bs,)
            rejected = outputs["rejected_mean_scores"]

            # print("chosen--C", chosen)
            # print("rejected--C", rejected)
            # print("T chosen---C:", infoTensor(chosen))
            # print("T rejected---C:", infoTensor(rejected))
            ''' only ph2
            chosen--C tensor([-0.4812, -0.0686, -0.5049, -0.2944, -0.4268, -0.7700, -0.4253, -0.3943],
                   device='cuda:0', dtype=torch.float16)
            rejected--C tensor([-0.4812, -0.0686, -0.5049, -0.2944, -0.4268, -0.7700, -0.4253, -0.3943],
                   device='cuda:1', dtype=torch.float16)
            T chosen---C: _Size([8])_float16_cuda:1_
            T rejected---C: _Size([8])_float16_cuda:1_
            '''

            #赋分正确"即为chosen分值大于rejected分值
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]

            # only ph2
            # print("correct_predictions--C", infoTensor(correct_predictions))
            # correct_predictions--C _Size([])_int64_cuda:0_  only ph2

            #print("total_predictions--C", total_predictions)  #total_predictions--C 672

            #累加每个step的平均chosen分值
            scores += outputs["chosen_mean_scores"].mean().float()

            if step == 99:  # For faster evaluation and debugging
                break

        # 计算acc指标
        acc = correct_predictions / total_predictions

        #计算当前step的平均chosen分值
        scores = scores / (step + 1)
        try:
            # 多进程结果求和求平均
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
        except:
            pass
        print("scores:", scores)
        print("acc:", acc)
        return scores, acc

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay)
    # print("optimizer_grouped_parameters :", optimizer_grouped_parameters)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    # print("AdamOptimizer :", AdamOptimizer)
    # AdamOptimizer : <class 'deepspeed.ops.adam.fused_adam.FusedAdam'>

    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))
    # print("optimizer :", optimizer)
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

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    print("num_update_steps_per_epoch :", num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    # print("lr_scheduler---2 :", lr_scheduler)
    # lr_scheduler---2 : <torch.optim.lr_scheduler.LambdaLR object at 0x7f1c900786a0>

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    # print("rm_model---4 :", rm_model)
    # print("optimizer---4 :", optimizer)
    # print("lr_scheduler---4 :", lr_scheduler)
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

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
        args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        print(" :", )

        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            # print容易引起打错过多的显示错位！！！
            # print_rank_0("batch :", batch)
            # print_rank_0("outputs :", outputs)
            # print_rank_0("loss :", loss)
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
            # print("T batch['input_ids']--D:", infoTensor(batch['input_ids']))
            # print("T batch['attention_mask']--D:", infoTensor(batch['attention_mask']))
            # print("T outputs['loss']--D:", infoTensor(outputs['loss']))
            # print("T outputs['chosen_mean_scores']--D:", infoTensor(outputs['chosen_mean_scores']))
            # print("T outputs['rejected_mean_scores']--D:", infoTensor(outputs['rejected_mean_scores']))
            # print("T loss--D:", infoTensor(loss))
            '''
            T batch['input_ids']--D: _Size([16, 128])_int64_cuda:1_
            T batch['attention_mask']--D: _Size([16, 128])_int64_cuda:1_
            T outputs['loss']--D: _Size([])_float16_cuda:1_
            T outputs['chosen_mean_scores']--D: _Size([8])_float16_cuda:1_
            T outputs['rejected_mean_scores']--D: _Size([8])_float16_cuda:1_
            T loss--D: _Size([])_float16_cuda:1_
            '''

            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)
        # Evaluate reward_loss on the validation set.
        print_rank_0(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****",
            args.global_rank)
        reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
        print_rank_0(
            f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
            args.global_rank)
        rm_model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print_rank_0('saving model ...', args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)
        # print("rm_model---5 :", rm_model)
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

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)

        if args.zero_stage == 3:
            debuginfo(prj='ds-chat')
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(rm_model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)


if __name__ == "__main__":
    main()
