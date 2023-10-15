# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from .reward_model import RewardModel

from pydebug import debuginfo, infoTensor

"""

1.3.1 基座模型结构
从基座模型的载入类可以大致知晓模型的结构，可见下方代码块。
此处使用了transformers.AutoModelForCausalLM.from_pretrained()来进行模型构建，
因此第一阶段的SFT（ref）模型将会是一个因果语言模型/自回归语言模型（CausalLM），
其所需要训练的任务自然就是自回归语言建模，即

003.png

模型调用create_hf_model方法进行构建,
参数指定有AutoModelForCausalLM
"""
# 2.2 模型训练
# 构建一个用于SFT训练的模型，模型可以指定为AutoModelForCausalLM类

# 这段代码定义了一个名为create_hf_model的函数，该函数的作用是创建或加载一个预训练模型。该函数的主要参数包括：
# model_class：模型的类别，例如GPT-2、BERT等。
# tokenizer：用于模型的分词器。
# ds_config: DeepSpeed的配置参数。
# rlhf_training：一个标志，用来表示是否正在进行RLHF（Reinforcement Learning from Human Feedback）训练。
# disable_dropout：一个标志，用来表示是否禁用dropout。Dropout是一种防止过拟合的技术。
def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    disable_dropout=False):
    # 根据model_name_or_path从预训练模型获取模型配置model_config。
	# 从预训练模型的路径或名称中加载模型配置
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    #print("model_config is", model_config)
    '''
    model_config is OPTConfig {
                  "_name_or_path": "/home/amd00/hf_model/opt-125m",
                  "_remove_final_layer_norm": false,
                  "activation_dropout": 0.0,
                  "activation_function": "relu",
                  "architectures": [
                    "OPTForCausalLM"
                  ],
                  "attention_dropout": 0.0,
                  "bos_token_id": 2,
                  "do_layer_norm_before": true,
                  "dropout": 0.1,
                  "enable_bias": true,
                  "eos_token_id": 2,
                  "ffn_dim": 3072,
                  "hidden_size": 768,
                  "init_std": 0.02,
                  "layer_norm_elementwise_affine": true,
                  "layerdrop": 0.0,
                  "max_position_embeddings": 2048,
                  "model_type": "opt",
                  "num_attention_heads": 12,
                  "num_hidden_layers": 12,
                  "pad_token_id": 1,
                  "prefix": "</s>",
                  "torch_dtype": "float16",
                  "transformers_version": "4.32.1",
                  "use_cache": true,
                  "vocab_size": 50272,
                  "word_embed_proj_dim": 768
                }

    '''

    # 如果disable_dropout为真，则将模型配置中的dropout设为0.0。
    if disable_dropout:
        # 禁用dropout
        # dropout是一种常见的正则化技术，它在训练过程中随机将部分神经元的输出设为0，有助于防止过拟合。
        model_config.dropout = 0.0

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    # 根据ds_config中的设置，创建DeepSpeed的配置对象dschf，以便进行DeepSpeed优化。
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        # ZeRO阶段3优化
        dschf = HfDeepSpeedConfig(ds_config)
        print("dschf is", dschf)
    else:
        dschf = None

    # 是否进行强化学习训练
    # 根据rlhf_training的值，确定是从配置中创建模型还是从预训练模型中加载模型。
	# 如果rlhf_training为真，则根据模型配置创建模型；否则，从预训练模型加载模型。
    if rlhf_training:
        debuginfo(prj="ds-chat", info = "rlhf_training")

        # 将使用模型配置（而非预训练权重）来创建模型
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
        # print("model-A is", model)
    else:
        debuginfo(prj="ds-chat", info="rlhf_training")

        # 将从预训练模型中加载权重
        # 如果模型的路径或名称包含".ckpt"，那么模型将从TensorFlow checkpoint加载权重。
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)
        # print("model-B is", model)

    # 将模型的结束标记和填充标记设为分词器的结束标记id。
	# 将模型配置中的结束符号id和填充符号id设为tokenizer的结束符号id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # 调整模型的词汇表大小，使其为8的倍数。这样做的目的是为了在某些硬件（如GPU）上提高效率。
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    #按照Causal Language Modeling进行训练，例如GPT、OPT、LLaMA、BLOOM等。
    # print("model-C is", model)

    return model

# 创建critic模型，该模型用于强化学习，通常评估一个给定action的价值。
def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        disable_dropout=False):
    
    # 1. 创建一个HuggingFace模型
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    """此处的模型读取方法用的是“AutoModel”，因此此处critic_model只有主干部分"""
    critic_model = create_hf_model(AutoModel, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training, disable_dropout)
    # print("critic_model-A is", critic_model)
    
	# 2. 在强化学习中评估动作的回报值
    """
       critic_model传入RewardModel，将额外得到线性层输出头，
       因此此处的critic_model结构为“v_head + 主干部分”
    """
    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)
    #print("critic_model-B is", critic_model)

    # 在RLHF训练模式下，为critic model加载预训练权重，以便在后续的训练过程中用于评估生成模型的表现。
    if rlhf_training:
        # 如果model_name_or_path不是一个目录（即它是一个模型的URL），则使用snapshot_download函数下载该模型。
        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)

        # critic model needs to load the weight here
        model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        # 使用torch.load函数加载模型权重，并用这些权重更新critic_model的状态字典。
        # map_location='cpu'参数确保模型权重被加载到CPU上。
        critic_model.load_state_dict(
            torch.load(model_ckpt_path, map_location='cpu'))

    #print("critic_model-C is", critic_model)

    return critic_model


'''
    model-C is OPTForCausalLM(
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
    model-A is OPTModel(
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
'''

'''
    critic_model-A is OPTModel(
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
        
        critic_model-B is RewardModel(
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
        
        critic_model-C is RewardModel(
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

model-B is OPTForCausalLM(
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