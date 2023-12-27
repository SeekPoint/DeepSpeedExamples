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

import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.utils import mem_estimate_log
from pydebug import gd, infoTensor

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
    gd.debuginfo(prj="ds_chat", info=f"model_config={model_config}")

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
        # gd.debuginfo(prj="ds_chat", info=f"dschf={dschf}")
        # dschf=<transformers.deepspeed.HfDeepSpeedConfig object at 0x7ff5713af5b0>
    else:
        dschf = None

    # 是否进行强化学习训练
    # 根据rlhf_training的值，确定是从配置中创建模型还是从预训练模型中加载模型。后面有补充区别
    if rlhf_training:
        gd.debuginfo(prj="ds_chat", info=f"将使用模型配置（而非预训练权重）来创建模型")

        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
        gd.debuginfo(prj="ds_chat", info=f"model-A={model}")
    else:
        gd.debuginfo(prj="ds_chat", info=f"将从预训练模型中加载模型及其权重")
        # 如果模型的路径或名称包含".ckpt"，那么模型将从tf checkpoint加载权重。
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config)
        gd.debuginfo(prj="ds_chat", info=f"model-B={model}")

    mem_estimate_log(args=None, exstr = '-0', model=model, num_gpus_per_node=2, num_nodes=1)

    gd.debuginfo(prj="ds_chat", info=f"tokenizer.eos_token_id is::, {tokenizer.eos_token_id}")
    gd.debuginfo(prj="ds_chat", info=f"model.config.eos_token_id={model.config.eos_token_id}")
    # 将模型配置中的结束符号id和填充符号id设为tokenizer的结束符号id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    #不能放在上面
    gd.debuginfo(prj="ds_chat", info=f"model.config.end_token_id is::, {model.config.end_token_id}")
    gd.debuginfo(prj="ds_chat", info=f"model.config.pad_token_id is::, {model.config.pad_token_id}")

    # 调整模型的词汇表大小，使其为8的倍数 yknote????为了在某些硬件（如GPU）上提高效率。
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    #按照Causal Language Modeling进行训练，例如GPT、OPT、LLaMA、BLOOM等。
    gd.debuginfo(prj="ds_chat", info=f"model-C={model}")

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
    gd.debuginfo(prj="ds_chat", info=f"critic_model-A={critic_model}")

    mem_estimate_log(args=None, exstr = 'ph2-0', model=critic_model, num_gpus_per_node=2, num_nodes=1)
    
    # 2. 在强化学习中评估动作的回报值
    # critic_model传入RewardModel进行改造！！
    # 将额外得到线性层输出头，因此此处的critic_model结构为“v_head + 主干部分”
    critic_model = RewardModel(
        critic_model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning)
    gd.debuginfo(prj="ds_chat", info=f"critic_model-B={critic_model}")

    mem_estimate_log(args=None, exstr = 'ph2-1', model=critic_model, num_gpus_per_node=2, num_nodes=1)

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

    gd.debuginfo(prj="ds_chat", info=f"critic_model-C={critic_model}")

    return critic_model

'''补充知识
https://stackoverflow.com/questions/72695297/difference-between-from-config-and-from-pretrained-in-huggingface

The two functions you described, from_config and from_pretrained, do not behave the same. 
For a model M, with a reference R:

    from_config allows you to instantiate a blank model,
         which has the same configuration (the same shape) as your model of choice: 
         M is as R was before training
         
    from_pretrained allows you to load a pretrained model,
         which has already been trained on a specific dataset for a given number of epochs: 
         M is as R after training.
'''

#ph-1