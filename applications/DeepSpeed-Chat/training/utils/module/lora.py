# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import math
import torch
from torch import nn
import torch.nn.functional as F
from deepspeed.compression.helper import recursive_getattr, recursive_setattr
import deepspeed

from pydebug import gd, infoTensor

'''
经过LoRA改造后，原始基座模型（此处的基座模型为“facebook/opt-125m”）的结构如下所示，
可见模型中除了输出头部分的线性层基本都被改成了LoRA结构，
因此模型在进行正向传播时也将流经LinearLayer_LoRA(nn.Module)中所定义的forward()方法（见上方代码块forward()部分）。

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

'''
# LinearLayer_LoRA 是一个实现了 LoRA（Low-Rank Adaptation）策略的 PyTorch 模块。
# LoRA 是一种在大型预训练模型（如 GPT-3 或者 BERT）微调中使用的策略，可以提高微调的效果。
# 这个类是一个简单的 LoRA 实现，只支持线性层。
class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    """具体的lora层"""
    # 初始化函数，定义了一些参数，如权重（weight）、LoRA 维度（lora_dim）、
    # LoRA 缩放因子（lora_scaling）、LoRA dropout （lora_droppout）以及偏置（bias）。
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None):
        gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")
        super(LinearLayer_LoRA, self).__init__()

        """此处的weight和bias即为原始结构中的参数"""
        self.weight = weight
        self.bias = bias

        # 在初始化过程中，首先检查 lora_dim 是否大于0，如果不是，就抛出异常。
        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        # 然后尝试获取权重的形状，如果权重是在 ZeRO 阶段3中的参数，那么通过 ds_shape 属性获取形状；
        # 否则直接使用 shape 属性。
        try:
            # for zero stage 3
            # 使用了DeepSpeed的阶段3优化
            # 在阶段3中，模型的权重会被切分成多个片段并分布在不同的设备上，因此无法直接获取权重的形状。
            rows, columns = weight.ds_shape
            gd.debuginfo(prj="ds_chat", info=f"z3: rows:{rows} ++ columns:{columns}")
        except:
            # 没有使用z3
            rows, columns = weight.shape
            gd.debuginfo(prj="ds_chat", info=f"Not z3: rows:{rows} ++ columns:{columns}")

        # 然后初始化 LoRA 的左右两个权重矩阵，并设置它们为可学习的参数。
        # 新的权重参数，shape = [columns, lora_dim]
        # 用于在低维空间中对原始权重进行适应(adapter)
        self.lora_right_weight = nn.Parameter(
            torch.zeros(columns, lora_dim))  # apply transpose so in forward we do not need to

        # 新的权重参数，shape = [lora_dim, rows]
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))

        # gd.debuginfo(prj = "ds_chat", info = f"self.lora_right_weight={self.lora_right_weight}")
        # gd.debuginfo(prj = "ds_chat", info = f"self.lora_left_weight={self.lora_left_weight}")
        gd.debuginfo(prj = "ds_chat", info = f"T self.lora_right_weight={infoTensor(self.lora_right_weight)}")
        gd.debuginfo(prj = "ds_chat", info = f"T self.lora_left_weight={infoTensor(self.lora_left_weight)}")

        # 缩放因子，用于调整LoRA参数的影响大小
        self.lora_scaling = lora_scaling / lora_dim
        gd.debuginfo(prj="ds_chat", info=f"self.lora_scaling: {self.lora_scaling}")

        # 如果 lora_droppout 大于0，则创建一个 Dropout 层；否则创建一个 Identity 层。
		# lora_dropout是在LoRA层的输出上应用的dropout
        if lora_droppout > 0:
            # 随机忽略一部分神经元（设置为0），使模型不能过分依赖任何一个神经元，提高模型的泛化能力。
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            # 一个恒等映射，不对输入做任何改变。
            self.lora_dropout = nn.Identity()

        # 然后调用 reset_parameters 方法对 LoRA 权重进行初始化。
        """初始化LoRA线性层的参数"""
        self.reset_parameters()

        # disable the original weight gradient
        """冻结weight部分的参数"""
        # 最后，关闭原始权重的梯度，设置 LoRA 融合标志位为 False。
		# 关闭原始权重的梯度更新
        self.weight.requires_grad = False

        # 是否已经将LoRA参数融合到原始权重中
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")

        # 将dropout层设置为评估模式
        # 这意味着在评估或测试过程中，dropout层不会进行任何操作，而是简单地将输入传递给下一个层。
        # 将模型设置为评估模式，这时候 Dropout 层会停止工作。
        self.lora_dropout.eval()

        # self.fuse_lora_weight()

    def train(self, mode=True):
        gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")
        # 在模型进行训练时调用
        # 它将dropout层设置为训练模式，这意味着在训练过程中，
        # dropout层会按照预设的概率随机地关闭输入中的部分元素，以防止过拟合。
        # 将模型设置为训练模式，这时候 Dropout 层会开始工作。
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    # 初始化 LoRA 权重的方法。右权重使用 kaiming 均匀分布进行初始化，左权重初始化为全0。
    def reset_parameters(self):
        gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")
        """初始化LoRA线性层的参数"""
        # 降维矩阵使用kaiming均匀分布初始化，
        # 服从均匀分布U(-\sqrt{1/in_feature}, +\sqrt{1/in_feature})
        # 与LoRA原始定义所用的(0,\sigma^2)正态分布初始化不同
		# 在创建LoRA层时会被调用, 初始化LoRA的权重参数
        # 使用了kaiming均匀分布进行初始化，这种方法在ReLU激活函数之后的线性层中表现较好。
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))

        # 升维矩阵使用全0初始化
        nn.init.zeros_(self.lora_left_weight)

    # fuse_lora_weight(self) 和 unfuse_lora_weight(self)：
    # 这两个方法用于将 LoRA 权重融合到原始权重中，或者从原始权重中解融合。
    # 融合操作实质上是将原始权重与 LoRA 权重的乘积（缩放后）相加。
    def fuse_lora_weight(self):
        gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")
        '''用于将原始的权重和LoRA的权重进行融合'''
        # 如果没有融合，那么它会将LoRA的权重和原始的权重加在一起。
        # 这个过程实际上是一个矩阵乘法操作，然后乘以一个比例因子lora_scaling。
        # 然后，它将fuse_lora设置为True，表示已经进行了权重融合。
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())

        self.fuse_lora = True

    def unfuse_lora_weight(self):
        gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")
        '''用于将融合后的权重分离开来'''
        # 如果已经进行了融合，那么它会将LoRA的权重从融合后的权重中减去，这样就得到了原始的权重。
        # 这个过程实际上是一个矩阵乘法操作，然后乘以一个比例因子lora_scaling。
        # 然后，它将fuse_lora设置为False，表示权重已经被分离。
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())

        self.fuse_lora = False

    '''
    LoRA正向传播
    常规部分的正向传播由transformers所定义，
    而LoRA部分的正向传播则由LinearLayer_LoRA(nn.Module)的forward()所定义（可见下方代码块），
    即“LoRA层的两条分支结果进行加和”。
    在代码中体现为
    F.linear(input, self.weight, self.bias) + 
    (self.lora_dropout(input) @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling，
    加号左侧为原结构支路，加号右侧为新增支路，self.lora_right_weight和self.lora_left_weight分别为两个新引入线性层的参数。
    '''
    # 前向传播函数。如果 LoRA 权重已融合，则直接对输入进行线性变换；
    # 否则，会额外计算一个 LoRA 项，该项是输入通过 Dropout 层，然后与 LoRA 权重相乘得到的。
    def forward(self, input):
        if self.fuse_lora:
            gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")
            # 如果fuse_lora为真，就使用融合后的权重进行线性变换，然后返回结果
            return F.linear(input, self.weight, self.bias)
        else:
            gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")
            """LoRA的正向传播"""
            # 否则，就分别用原始权重和LoRA权重进行线性变换，将两个结果加在一起，然后返回
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling

# 这个函数 convert_linear_layer_to_lora 是用来将模型中的线性层转换为 LoRA 层的。
# 在训练深度学习模型时，这种方法能够在保持预训练模型参数不变的同时，通过添加额外的参数来微调模型。
# convert the linear layer to LoRA
# 将模型中指定的线性层转换为LoRA（Low-Rank Adaptation）层
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_droppout=0):
    # 保存需要转换为LoRA层的模块的名称
    """
    将名称中带有"decoder.layers."的线性层转换为lora层
    """
    """取出模型中参数名含有decoder.layers.的线性层"""
    repalce_name = []

    # 函数遍历repalce_name列表中的每一个名称
    # 函数首先遍历模型中的所有模块（model.named_modules()），找出名称中包含 part_module_name 的线性层（nn.Linear），
    # 并将这些层的名称添加到 repalce_name 列表中。
    for name, module in model.named_modules():
        gd.debuginfo(prj="ds_chat", info=f"name={name} +++ module={module}")
        if isinstance(module, nn.Linear) and part_module_name in name:
            repalce_name.append(name)

   # gd.debuginfo(prj="ds_chat", info=f"ALL repalce_name is: {repalce_name}")

    # 然后，函数遍历 repalce_name 列表，使用 recursive_getattr 函数获取模型中对应名称的模块。
    # 这些模块是需要被替换成 LoRA 层的线性层。
    for name in repalce_name:
        """recursive_getattr实现了从model中根据属性名取出对应原始结构"""
        # 获取模型中对应的模块
        module = recursive_getattr(model, name)
        gd.debuginfo(prj="ds_chat", info=f"name={name} +++ modele={module}")

        # 使用LinearLayer_LoRA类创建一个新的LoRA层，该层的权重、偏置以及其他参数从原模块中继承，
        # 并且将其移到了原模块的设备和数据类型上。
        """纳入原始结构的参数，实例化lora层"""
        # 对于每一个需要被替换的模块，函数创建一个 LinearLayer_LoRA 实例 tmp，
        # 并将其传输到与原始模块相同的设备和数据类型上。创建 LinearLayer_LoRA 实例时，
        # 需要传入原始模块的权重、偏置以及 LoRA 层的一些参数，如 lora_dim、lora_scaling 和 lora_droppout。
        tmp = LinearLayer_LoRA(
            module.weight, lora_dim, lora_scaling, lora_droppout,
            module.bias).to(module.weight.device).to(module.weight.dtype)

        gd.debuginfo(prj="ds_chat", info=f"tmp module is: {tmp}")

        # 将模型中的原模块替换为新的LoRA层
        """recursive_getattr实现了将model对应属性的结构换成lora层实例"""
        # 创建完 LinearLayer_LoRA 实例后，函数使用 recursive_setattr 函数将原始模块替换为 LinearLayer_LoRA 实例。
        recursive_setattr(model, name, tmp)

    return model

# 这个函数的主要功能是筛选出那些在DeepSpeed Zero 3优化中被离线存储，但在当前还未获取的参数。
# 在DeepSpeed Zero 3优化中，一些模型参数在使用过后会被离线存储，以此释放GPU显存。
# 当这些参数需要再次被使用时，需要先获取到本地。
# 从给定的参数列表param_list中获取那些在当前GPU上不可用的ZeRO-3分区参数
def _z3_params_to_fetch(param_list):
    # yknote--代码有改动
	# 这个条件语句判断一个参数是否是被DeepSpeed Zero 3优化过的，且其状态为"未获取"（NOT_AVAILABLE）。
    # 对于被DeepSpeed Zero 3优化过的参数，它们有一个ds_id属性和一个ds_status属性，其中ds_status表示参数的当前状态。
    # 检查每个参数是否有ds_id属性，这是DeepSpeed的ZeRO-3优化的标记，如果一个参数有这个属性，那么它被ZeRO-3分区。
    # 检查该参数的ds_status属性，表示参数在当前设备上是否可用的标志。如果等于NOT_AVAILABLE，则表示该参数在当前设备上不可用。
    # return [
    #     p for p in param_list
    #     if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.
    #     partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    # ]
    
    #yknote改写
    tmp = []
    for p in param_list:
        gd.debuginfo(prj="ds_chat", info=f"p={infoTensor(p)}")
        if hasattr(p, 'ds_id') and p.ds_status == deepspeed.runtime.zero.partition_parameters.ZeroParamStatus.NOT_AVAILABLE:
            tmp.append(p)
    gd.debuginfo(prj="ds_chat", info=f"tmp={tmp}")
    return tmp
            

# 这个函数 convert_lora_to_linear_layer 是用来将模型中的 LoRA 层转换回线性层的。
# 在训练深度学习模型时，这个操作可以用于在训练完 LoRA 层后，将模型恢复到原始的状态，
# 以便进行下一步的操作，如模型的保存、加载等。
# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    repalce_name = []

    # 函数首先遍历模型中的所有模块（model.named_modules()），找出所有的 LoRA 层（LinearLayer_LoRA），
    # 并将这些层的名称添加到 repalce_name 列表中。
    for name, module in model.named_modules():
        gd.debuginfo(prj="ds_chat", info=f"name={name} +++ module={module}")
        # 如果某个模块是LoRA层，那么将其名字添加到repalce_name列表中。
        if isinstance(module, LinearLayer_LoRA):
            repalce_name.append(name)

    # gd.debuginfo(prj="ds_chat", info=f"ALL repalce_name is: {repalce_name}")

    # 然后，函数遍历 repalce_name 列表，使用 recursive_getattr 函数获取模型中对应名称的 LoRA 层
    for name in repalce_name:
        # 获取对应的模块
        module = recursive_getattr(model, name)
        gd.debuginfo(prj="ds_chat", info=f"name={name} +++ module={module}")

        # 对于每一个 LoRA 层，函数首先检查是否处于 zero stage 3（DeepSpeed 的一个特性，用于在多GPU训练中减少内存占用）。
        # 如果是，则设置 zero_stage_3 为 True。这里就是检测是否使用了DeepSpeed的ZeRO-3优化策略
        zero_stage_3 = hasattr(module.weight, 'ds_id')

        # 获取LoRA层中的所有参数
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([
                module.weight, # 权重
                module.bias,  # 偏置
                module.lora_left_weight, # 左侧权重
                module.lora_right_weight # 右侧权重
        ]),
                modifier_rank=0,
                enabled=zero_stage_3):

            # 将LoRA层转换成普通的全连接层
            module.fuse_lora_weight()

    return model

# 这个函数的作用是关闭模型中除LoRA参数之外的所有参数的梯度。这意味着在训练过程中，只有LoRA参数会被优化，其他参数保持不变。
def only_optimize_lora_parameters(model):
    gd.debuginfo(prj="ds_chat")
    # turn off the gradient of all the parameters except the LoRA parameters
    # 遍历模型的所有参数，每个参数都有一个唯一的名称name和对应的参数值param。
    # 目标是让模型只优化LoRA参数， 获取模型的所有参数及其名称
    for name, param in model.named_parameters():
        # 查当前参数的名称是否包含lora_right_weight或lora_left_weight。
        # 这是因为在LoRA（Low-Rank Adaptation）中，只有这两种参数是需要优化的。
		# 如果参数名称中含有"lora_right_weight"或"lora_left_weight"（这是LoRA层中权重的参数名），
        # 就将该参数的requires_grad属性设置为True，使得该参数在接下来的训练中可以被优化。
        if "lora_right_weight" in name or "lora_left_weight" in name:
            gd.debuginfo(prj="ds_chat", info=f"name={name} +++ param={param} set requires_grad=true")
            # param.requires_grad = True 如果参数名包含lora_right_weight或lora_left_weight，
            # 则设置参数的requires_grad属性为True，表示需要对此参数进行梯度下降优化。
            param.requires_grad = True
        else:
            gd.debuginfo(prj="ds_chat", info=f"name={name} +++ param={param} set requires_grad=false")
            # 否则，不被优化
            param.requires_grad = False
    return model

'''
self.lora_right_weight : Parameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.]], requires_grad=True)

self.lora_left_weight : Parameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.]], requires_grad=True)


ds_chat pid: 92669 at MZ32-00 File: training/utils/module/lora.py Func: __init__ Line#: 109 Info: self.lora_right_weight : Parameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], requires_grad=True)
ds_chat pid: 92669 at MZ32-00 File: training/utils/module/lora.py Func: __init__ Line#: 110 Info: self.lora_left_weight : Parameter containing:
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]], requires_grad=True)
'''