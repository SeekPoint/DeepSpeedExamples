# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler

from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model, create_critic_model
from utils.utils import get_optimizer_grouped_parameters

from pydebug import gd, infoTensor

"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


def log_init(model_name, stime=None):
    '''在模型初始化开始和结束时打印日志的函数
    model_name : 被初始化的模型的名称
    stime : 开始时间
    '''
    # 检查当前的分布式计算rank，只有rank为0的进程才会打印日志，
    # 这是为了避免在分布式环境中每个进程都打印日志，导致日志重复和混乱。
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            # 计算初始化耗费的时间，并将其格式化为字符串
            duration = "(duration: {:.2f}s)".format(time.time() - stime)

        # 日志信息 msg
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"

        # 计算日志字符串两边需要打印的星号(*)数量，以使得整个日志信息长度为90
        stars = (90 - len(msg)) // 2

        # 打印出由星号、日志信息和星号组成的字符串
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()

'''
3.3.2 初始化各相关模型
3.3.2.1 模型初始化过程
源码中使用了 DeepSpeedRLHFEngine类进行了actor、ref/SFT、critic、reward/RM、actor_ema等模型的初始化，
该类主要实现了：

    1读取模型，虽然同样支持直接从huggingface hub拉取相应模型，但通常都是从本地路径读取phase1、phase2训练好的模型：
    
        1actor、ref/SFT以及actor_ema（如果开启了ema）通常都初始化自phase1训练所得的模型；
        
        2critic与reward通常都初始化自phase2训练所得的模型。
        
    2为各个相关模型设定不同的DeepSpeed配置（ds_config）并使用DeepSpeedEngine进行封装，
    而actor默认情况下将使用DeepSpeedHybridEngine进行封装，DeepSpeedHybridEngine的简单介绍可见下方；
    
    3最终得到1个携有所有相关模型的对象rlhf_engine。

模型初始化的相关代码
'''
class DeepSpeedRLHFEngine():

    def __init__(self, actor_model_name_or_path, critic_model_name_or_path,
                 tokenizer, args, num_total_iters):
        """
        加载模型并进行DS封装
        1. actor与ref（以及actor_ema）通常都初始化自phase1训练所得的模型；
        2. critic与reward通常都初始化自phase2训练所得的模型。
        根据它们的入参就能知道。
        """
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer

        # 用训练好的SFT模型初始化Actor模型
        # 此处的actor是模型经过DeepSpeed封装后得到的DeepSpeedHybridEngine对象
        self.actor = self._init_actor(
            actor_model_name_or_path=actor_model_name_or_path)

        print("self.actor is:", self.actor)

        # 用训练好的SFT模型初始化SFT模型
        #此处的reference是模型经过DeepSpeed封装后得到的DeepSpeedEngine对象
        self.ref = self._init_ref(
            actor_model_name_or_path=actor_model_name_or_path)
        # print("self.ref is:", self.ref)

        self.actor_ema = None

        #如果开启了ema，则初始化并封装ema
        if self.args.enable_ema:
            #此处的ema是模型经过DeepSpeed封装后得到的DeepSpeedEngine对象
            self.actor_ema = self._init_ema(
                actor_model_name_or_path=actor_model_name_or_path)

        # print("self.actor_ema is:", self.actor_ema)

        # 用训练好的RW初始化Critic模型
        # 此处的critic是模型经过DeepSpeed封装后得到的DeepSpeedEngine对象
        self.critic = self._init_critic(
            critic_model_name_or_path=critic_model_name_or_path)
        # print("self.critic is:", self.critic)


        # 用训练好的RW初始化reward模型
        # 此处的reward是模型经过DeepSpeed封装后得到的DeepSpeedEngine对象
        self.reward = self._init_reward(
            critic_model_name_or_path=critic_model_name_or_path)
        # print("self.reward is:", self.reward)

        if self.args.critic_gradient_checkpointing:
            self.critic.gradient_checkpointing_enable()

    # actor模型
    def _init_actor(self, actor_model_name_or_path):
        """
        初始化actor并使用DeepSpeedHybridEngine封装
        :param actor_model_name_or_path: phase1训练好的actor模型路径
        :return: 经DeepSpeedHybridEngine封装的actor


        DS Config
        根据传参构建ds config，
        与其他相关模型不同的地方在于，如果传参指定启用了enable_hybrid_engine，
        那么HybridEngine将作用于actor，对actor进行封装，
        因为HybridEngine可以使得模型可以在训练与推理两种模式中进行自动切换，
        同时享有训练与推理的优化，
        这对于既需要进行推理生成、又需要进行训练的actor来说是有增益作用的。
        """
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
		
        # 打印一条关于actor模型初始化开始的信息
        stime = log_init("Actor")

        # DS Config 获取训练配置
        ds_config = get_train_ds_config(
            offload=self.args.offload, # 是否启用offload模式
            stage=self.args.actor_zero_stage, # ZeRO的阶段
            enable_hybrid_engine=self.args.enable_hybrid_engine, # 是否启用混合引擎
            inference_tp_size=self.args.inference_tp_size, # 用于推理的张量切分的大小
            release_inference_cache=self.args.release_inference_cache, # 是否在推理完成后释放cache
            pin_parameters=(not self.args.unpin_actor_parameters), # 是否在CPU上固定模型参数
            tp_gather_partition_size=self.args.tp_gather_partition_size, # 聚合张量切分的大小
            max_out_tokens=self.args.max_prompt_seq_len + self.args.max_answer_seq_len, # 模型的最大输出序列长度
            enable_tensorboard=self.args.enable_tensorboard,
            tb_path=self.args.tensorboard_path,
            tb_name="step3_actor")
        print("_init_actor ds_config train---1:", ds_config)  # 一直打开

        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size  # 每个GPU的微批次训练大小
        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor # 全局训练批次大小

        print("_init_actor ds_config train ---2:", ds_config) #一直打开

        # Model : 创建 actor model
        # Model 使用CausalLM结构载入模型及权重，实例化actor
        actor_model = create_hf_model(
            model_class=AutoModelForCausalLM,
            model_name_or_path=actor_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_config,
            disable_dropout=self.args.disable_actor_dropout)

        print("s3 create_hf_model actor_model:", actor_model)

        # LoRA
        # 如果开启LoRA训练则添加LoRA旁路
        if self.args.actor_lora_dim > 0:
            # 在模型中找到指定的模块并将其全连接层转换为LoRA层
            actor_model = convert_linear_layer_to_lora(
                actor_model, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)

            print("s3 convert_linear_layer_to_lora actor_model:", actor_model)

            if self.args.only_optimize_lora:
                # 只有LoRA层的参数会被更新，而其他层的参数将保持不变
                actor_model = only_optimize_lora_parameters(actor_model)
                print("s3 only_optimize_lora_parameters actor_model:", actor_model)


        # Optimizer
        # 实例化优化器：分组权重衰减等
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam

        # 将模型参数按照是否应用权重衰减进行分组
        optim_params = get_optimizer_grouped_parameters(
            actor_model, self.args.actor_weight_decay)

        # 使用Adam优化器初始化优化器实例
        optim = AdamOptimizer(optim_params,
                              lr=self.args.actor_learning_rate,
                              betas=(0.9, 0.95)) # beta参数决定了梯度和梯度平方的滑动平均值的更新速度

        # LR Scheduler
        # 实例化学习率调度器
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type, # 学习率调度器的类型
            optimizer=optim, # 优化器实例
            num_warmup_steps=self.args.num_warmup_steps, # 在学习率调度过程中，预热阶段的步数
            num_training_steps=self.num_total_iters, # 总的训练步数
        )

        # DeepSpeed Engine
        # DeepSpeedEngine封装
        # 若ds_config中定义了启用HybridEngine，
        # 则返回的actor_engine不仅是个DeepSpeedEngine实例，
        # 确切地说还是个DeepSpeedHybridEngine实例，集成有HybridEngine的优化

        #TODO: move enable_hybrid_engine and pin_parameters to ds_config
        print("#######ph3 actor_model deepspeed.initialize ################################################")
        actor_engine, *_ = deepspeed.initialize(model=actor_model, # 需要训练的模型
                                                optimizer=optim, # 优化器
                                                lr_scheduler=lr_scheduler, # 学习率调度器
                                                config=ds_config # 设置DeepSpeed引擎的配置
                                                )
        print("#######ph3 actor_model deepspeed.initialize ################################################")

        log_init("Actor", stime=stime)

        # print("optim_params is:", optim_params)
        # print("optim is:", optim)
        # print("lr_scheduler is:", lr_scheduler)

        print("actor_engine is:", actor_engine)

        return actor_engine

    """
    其余ref、actor_ema、critic、reward的初始化几乎同理，
    只是ds_config设置不同，但最终都将返回经DeepSpeedEngine封装的对象。
    """
    # ref模型
    def _init_ref(self, actor_model_name_or_path):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        '''初始化参考模型（Ref model）'''
        stime = log_init("Ref")

        # DS Config
        zero_stage = self.args.actor_zero_stage

        if zero_stage != 3:
            gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory for ref model
            # 如果Actor模型使用了ZeRO-3阶段，那么参考模型也使用，否则使用ZeRO-0阶段。
            # 区别：
            # ① ZeRO-3是最高级别的内存优化，需要的GPU内存最小，但计算开销较大；
            # ② ZeRO-0是不做内存优化，需要的GPU内存最大，但计算开销较小。
            zero_stage = 0

        # 定义DeepSpeed的配置 
        ds_config = get_eval_ds_config(self.args.offload_reference_model,zero_stage)
        print("_init_ref ds_config eval ---1:", ds_config)  # 一直打开
		
        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size

        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        print("_init_ref ds_config eval ---2:", ds_config)  # 一直打开

        # 创建模型
        ## 往往会再定义一个ref model，为原始的actor_model，用来计算KL避免生成的内容与原始模型差太远【怕训飞】。
        ## ref_model 不会进行参数更新
        ref_model = create_hf_model(AutoModelForCausalLM,
                                    actor_model_name_or_path, self.tokenizer,
                                    ds_config)

        print("s3 create_hf_model ref_model is:", ref_model)
									
        # DeepSpeed初始化
        # 参考模型不需要优化器和学习率调度器，所以在初始化DeepSpeed时只需要传入模型和配置即可。
        print("#######ph3 ref_model deepspeed.initialize ################################################")
        ref_engine, *_ = deepspeed.initialize(model=ref_model,
                                              config=ds_config)
        print("#######ph3 ref_model deepspeed.initialize ################################################")


        print("ref_engine is:", ref_engine)

        log_init("Ref", stime=stime)
        return ref_engine

    # ema模型
    def _init_ema(self, actor_model_name_or_path):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        '''初始化指数移动平均（Exponential Moving Average，EMA）模型
        知识补充:
        EMA模型是用来平滑模型权重的，它会根据设定的衰减率持续追踪模型的运动平均。
        在每一步训练中，它都会将一部分当前模型的参数和一部分EMA模型的参数相加，用来更新EMA模型的参数。
        这样，EMA模型就会保存一个当前模型参数的长期平均，这有助于提高模型的稳定性。
        '''
        stime = log_init("EMA")

        # DS Config
        zero_stage = self.args.actor_zero_stage

        # 是否启用了ZeRO-3，设置zero_stage
        if zero_stage != 3:
            # If actor is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0
			
        # 定义DeepSpeed的配置
        ds_config = get_eval_ds_config(self.args.offload_reference_model,
                                       zero_stage)
        print("_init_ema ds_config eval ---1:", ds_config)  # 一直打开

        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size

        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps_actor

        print("_init_ema ds_config eval ---2:", ds_config)  # 一直打开

        # 创建模型
        actor_model_ema = create_hf_model(AutoModelForCausalLM,
                                          actor_model_name_or_path,
                                          self.tokenizer, ds_config)

        print("s3 create_hf_model actor_model_ema:", actor_model_ema)

        if self.args.actor_lora_dim > 0:
            actor_model_ema = convert_linear_layer_to_lora(
                actor_model_ema, self.args.actor_lora_module_name,
                self.args.actor_lora_dim)
            print("s3 convert_linear_layer_to_lora actor_model_ema:", actor_model_ema)

        print("#######ph3 actor_model_ema deepspeed.initialize ################################################")
        ema_engine, *_ = deepspeed.initialize(model=actor_model_ema,
                                              config=ds_config)
        print("#######ph3 actor_model_ema deepspeed.initialize ################################################")

        print("ema_engine is:", ema_engine)

        log_init("EMA", stime=stime)

        return ema_engine

    # critic模型
    def _init_critic(self, critic_model_name_or_path):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        stime = log_init("Critic")

        ds_config = get_train_ds_config(
            offload=self.args.offload,
            stage=self.args.critic_zero_stage,
            enable_tensorboard=self.args.enable_tensorboard,
            tb_path=self.args.tensorboard_path,
            tb_name="step3_critic")

        print("_init_critic ds_config train ---1:", ds_config)  # 一直打开

        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size

        #TODO(jeff): we should probably set grad accumlation steps here as well for clarity
        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        print("_init_critic ds_config train ---2:", ds_config)  # 一直打开

        #TODO(jeff): should not be needed, we should be able to use ds_config above
        #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
        ds_eval_config = get_eval_ds_config(offload=False, stage=0)
        print("_init_critic ds_config eval:", ds_config)  # 一直打开

        # Model
        critic_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True,
            disable_dropout=self.args.disable_critic_dropout)
        print("s3 critic_model = create_critic_model:", critic_model)

        # LoRA
        if self.args.critic_lora_dim > 0:

            critic_model = convert_linear_layer_to_lora(
                critic_model, self.args.critic_lora_module_name,
                self.args.critic_lora_dim)

            print("s3 convert_linear_layer_to_lora critic_model:", critic_model)

            if self.args.only_optimize_lora:
                critic_model = only_optimize_lora_parameters(critic_model)
                print("s3 only_optimize_lora_parameters critic_model:", critic_model)

        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_pararms = get_optimizer_grouped_parameters(
            critic_model, self.args.critic_weight_decay)
        optim = AdamOptimizer(optim_pararms,
                              lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )

        # DeepSpeed Engine
        print("#######ph3 critic_model deepspeed.initialize ################################################")
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim,
                                                 lr_scheduler=lr_scheduler,
                                                 config=ds_config)
        print("#######ph3 critic_model deepspeed.initialize ################################################")

        log_init("Critic", stime=stime)
        return critic_engine

    # reward模型
    def _init_reward(self, critic_model_name_or_path):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        stime = log_init("Reward")

        # DS Config
        zero_stage = self.args.critic_zero_stage

        if zero_stage != 3:
            # If critic is ZeRO-3 then we use it for everything, otherwise assume we have enough memory
            zero_stage = 0

        ds_config = get_eval_ds_config(offload=self.args.offload,
                                       stage=zero_stage)
        print("_init_reward ds_config eval--1 :", ds_config)  # 一直打开

        ds_config[
            'train_micro_batch_size_per_gpu'] = self.args.per_device_mini_train_batch_size

        ds_config[
            'train_batch_size'] = self.args.per_device_mini_train_batch_size * torch.distributed.get_world_size(
            ) * self.args.gradient_accumulation_steps

        print("_init_reward ds_config eval--2:", ds_config)  # 一直打开

        #TODO(jeff): should not be needed, we should be able to use ds_config above
        #TODO(jeff): it means we never create the critic w. zero.init context if we are using ZeRO-3
        ds_eval_config = get_eval_ds_config(offload=False, stage=0)
        print("_init_reward eval ds_config --3:", ds_eval_config)  # 一直打开

        ## reward model 和 critic model 都是用step 2 的模型初始化，step 3 中 reward model 不再训练
        # Model
        reward_model = create_critic_model(
            model_name_or_path=critic_model_name_or_path,
            tokenizer=self.tokenizer,
            ds_config=ds_eval_config,
            num_padding_at_beginning=self.args.num_padding_at_beginning,
            rlhf_training=True)

        print("#######ph3 reward_engine deepspeed.initialize ################################################")
        reward_engine, *_ = deepspeed.initialize(model=reward_model,
                                                 config=ds_config)
        print("#######ph3 reward_engine deepspeed.initialize ################################################")

        log_init("Reward", stime=stime)
        return reward_engine


'''
ema_engine is: DeepSpeedEngine(
      (module): OPTForCausalLM(
        (model): OPTModel(
          (decoder): OPTDecoder(
            (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
    )
actor_model_ema is: OPTForCausalLM(
      (model): OPTModel(
        (decoder): OPTDecoder(
          (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
self.reward is: DeepSpeedEngine(
      (module): RewardModel(
        (v_head): Linear(in_features=768, out_features=1, bias=False)
        (rwtranrsformer): OPTModel(
          (decoder): OPTDecoder(
            (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
  _init_ref ds_config is: {'train_batch_size': 8, 'train_micro_batch_size_per_gpu': 4, 'steps_per_print': 10,
 'zero_optimization': {'stage': 0, 'stage3_param_persistence_threshold': 10000.0, 
 'offload_param': {'device': 'none'}, 'memory_efficient_linear': False}, 'fp16': {'enabled': True}, 
 'gradient_clipping': 1.0, 'prescale_gradients': False, 'wall_clock_breakdown': False}


ref_model is: OPTForCausalLM(
  (model): OPTModel(
    (decoder): OPTDecoder(
      (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
ref_engine is: DeepSpeedEngine(
  (module): OPTForCausalLM(
    (model): OPTModel(
      (decoder): OPTDecoder(
        (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
)
'''

'''
self.critic is: DeepSpeedEngine(
      (module): RewardModel(
        (v_head): Linear(in_features=768, out_features=1, bias=False)
        (rwtranrsformer): OPTModel(
          (decoder): OPTDecoder(
            (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
optim_params is: [{'params': [tensor([-0.0078, -0.0563, -0.0166,  ...,  0.3953,  0.3679,  0.5161],
        device='cuda:1', requires_grad=True)], 'weight_decay': 0.0, 'lr': 0.0, 'bias_correction': True, 'betas': (0.9, 0.95), 'eps': 1e-08, 'initial_lr': 9.65e-06, 'step': 0}, {'params': [tensor([ 0.0646,  0.0148,  0.0143,  ..., -0.0972, -0.0310, -0.0812],
        device='cuda:1', requires_grad=True)], 'weight_decay': 0.0, 'lr': 0.0, 'bias_correction': True, 'betas': (0.9, 0.95), 'eps': 1e-08, 'initial_lr': 9.65e-06, 'step': 0}]
optim is: FusedAdam (
Parameter Group 0
    betas: (0.9, 0.95)
    bias_correction: True
    eps: 1e-08
    initial_lr: 9.65e-06
    lr: 0.0
    step: 0
    weight_decay: 0.0

Parameter Group 1
    betas: (0.9, 0.95)
    bias_correction: True
    eps: 1e-08
    initial_lr: 9.65e-06
    lr: 0.0
    step: 0
    weight_decay: 0.0
)
lr_scheduler is: <torch.optim.lr_scheduler.LambdaLR object at 0x7f77d29c2e20>


actor_model is: OPTForCausalLM(
  (model): OPTModel(
    (decoder): OPTDecoder(
      (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
actor_engine is: DeepSpeedHybridEngine(
  (module): OPTForCausalLM(
    (model): OPTModel(
      (decoder): OPTDecoder(
        (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
)
'''

'''
self.ref is: DeepSpeedEngine(
      (module): OPTForCausalLM(
        (model): OPTModel(
          (decoder): OPTDecoder(
            (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
    )
'''

'''
self.actor_ema is: DeepSpeedEngine(
      (module): OPTForCausalLM(
        (model): OPTModel(
          (decoder): OPTDecoder(
            (embed_tokens): Embedding(50272, 768, padding_idx=2)
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
    )
'''

'''
_init_actor ds_config: {'train_batch_size': 8, 'train_micro_batch_size_per_gpu': 4, 'steps_per_print': 10, 
'zero_optimization': {'stage': 2, 'offload_param': {'device': 'none'}, 'offload_optimizer': {'device': 'none'}, 
'stage3_param_persistence_threshold': 10000.0, 'stage3_max_live_parameters': 30000000.0, 
'stage3_prefetch_bucket_size': 30000000.0, 'memory_efficient_linear': False}, 
'fp16': {'enabled': True, 'loss_scale_window': 100}, 'gradient_clipping': 1.0, 
'prescale_gradients': False, 'wall_clock_breakdown': False, 
'hybrid_engine': {'enabled': True, 'max_out_tokens': 512, 'inference_tp_size': 1, 
'release_inference_cache': False, 'pin_parameters': True, 'tp_gather_partition_size': 8}, 
'tensorboard': {'enabled': False, 'output_path': 'step3_tensorboard/ds_tensorboard_logs/', 
'job_name': 'step3_actor_tensorboard'}}
'''

'''
_init_ema ds_config is: {'train_batch_size': 8, 'train_micro_batch_size_per_gpu': 4, 'steps_per_print': 10,
 'zero_optimization': {'stage': 0, 'stage3_param_persistence_threshold': 10000.0, 
 'offload_param': {'device': 'none'}, 'memory_efficient_linear': False}, 'fp16': {'enabled': True}, 
 'gradient_clipping': 1.0, 'prescale_gradients': False, 'wall_clock_breakdown': False}
'''

'''
_init_reward ds_config is: {'train_batch_size': 8, 'train_micro_batch_size_per_gpu': 4, 
'steps_per_print': 10, 'zero_optimization': {'stage': 0, 'stage3_param_persistence_threshold': 10000.0, 
'offload_param': {'device': 'none'}, 'memory_efficient_linear': False}, 'fp16': {'enabled': True}, 
'gradient_clipping': 1.0, 'prescale_gradients': False, 'wall_clock_breakdown': False}
'''