# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
import deepspeed

'''
经过LoRA改造后，原始基座模型（此处的基座模型为“facebook/opt-125m”）的结构如下所示，可见模型中除了输出头部分的线性层基本都被改成了LoRA结构，因此模型在进行正向传播时也将流经LinearLayer_LoRA(nn.Module)中所定义的forward()方法（见上方代码块forward()部分）。

OPTForCausalLM(
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
————————————————
'''
class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    """具体的lora层"""
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None):
        super(LinearLayer_LoRA, self).__init__()

        """此处的weight和bias即为原始结构中的参数"""
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        """初始化LoRA线性层的参数"""
        self.reset_parameters()

        # disable the original weight gradient
        """冻结weight部分的参数"""
        self.weight.requires_grad = False

        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        """初始化LoRA线性层的参数"""
        # 降维矩阵使用kaiming均匀分布初始化，
        # 服从均匀分布U(-\sqrt{1/in_feature}, +\sqrt{1/in_feature})
        # 与LoRA原始定义所用的(0,\sigma^2)正态分布初始化不同
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))

        # 升维矩阵使用全0初始化
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    '''
    LoRA正向传播
    常规部分的正向传播由transformers所定义，
    而LoRA部分的正向传播则由LinearLayer_LoRA(nn.Module)的forward()所定义（可见下方代码块），
    即“LoRA层的两条分支结果进行加和”。
    在代码中体现为F.linear(input, self.weight, self.bias) + (self.lora_dropout(input) @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling，
    加号左侧为原结构支路，加号右侧为新增支路，self.lora_right_weight和self.lora_left_weight分别为两个新引入线性层的参数。
    
    '''
    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            """LoRA的正向传播"""
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling


# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_droppout=0):
    """
    将名称中带有"decoder.layers."的线性层转换为lora层
    """
    """取出模型中参数名含有decoder.layers.的线性层"""
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and part_module_name in name:
            repalce_name.append(name)
    for name in repalce_name:
        """recursive_getattr实现了从model中根据属性名取出对应原始结构"""
        module = recursive_getattr(model, name)
        """纳入原始结构的参数，实例化lora层"""
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)
        """recursive_getattr实现了将model对应属性的结构换成lora层实例"""
        recursive_setattr(model, name, tmp)
    return model


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
        partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]


# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            repalce_name.append(name)
    for name in repalce_name:
        module = recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, module.bias, module.lora_left_weight,
                module.lora_right_weight
        ]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()
    return model


def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model
