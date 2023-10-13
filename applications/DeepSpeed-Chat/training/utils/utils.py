# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from pydebug import debuginfo, infoTensor

# 在rank0也就是master rank打印信息，防止每个机器或GPU都打印消息造成大量重复信息
def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)

# 这个函数的作用是把一个数据批次移动到指定的设备上。在PyTorch中，数据默认是在CPU上的，
# 如果要在GPU上进行运算，就需要先把数据移到GPU上。这个函数通过遍历批次中的所有元素并
# 调用to(device)方法来实现这一点。如果某个元素不能被移到指定的设备上
#（例如，这个元素是一个字符串或者其他不能在GPU上使用的类型），那么就直接保留这个元素，不进行任何操作。
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


# 这段代码定义了一个名为load_hf_tokenizer的函数，该函数的作用是加载一个指定的分词器。这个函数的参数有：
# model_name_or_path：模型的名称或者预训练模型的路径。
# fast_tokenizer：是否使用Fast Tokenizer，这是一种更快的分词器实现。
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    # 检查model_name_or_path是否存在。如果存在，可能意味着这是一个本地的模型路径。
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        # 如果是本地模型，再检查在该路径下的"config.json"文件是否存在。如果存在，打开该文件，并从文件中读取"_name_or_path"字段，将其赋值给model_name，然后通过AutoTokenizer.from_pretrained函数，使用该模型名称和fast_tokenizer参数，从HuggingFace模型库下载并加载分词器。
        model_json = os.path.join(model_name_or_path, "config.json")
        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, fast_tokenizer=fast_tokenizer)
    else:
        # 如果model_name_or_path不存在，直接使用AutoTokenizer.from_pretrained函数，使用model_name_or_path和fast_tokenizer参数，从HuggingFace模型库下载并加载分词器。
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
    return tokenizer


# 这段代码的功能是将训练好的模型以Hugging Face格式保存，这样之后就可以使用Hugging Face库的from_pretrained方法加载了。
def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    # 首先，这行代码检查模型是否有'module'这个属性。这在PyTorch中是一个常见的模式，
    # 当模型被封装在torch.nn.DataParallel或torch.nn.parallel.DistributedDataParallel时，
    # 模型的所有属性都会被存储在'module'属性中。所以这行代码的目的是确保我们总是在原始模型上进行操作，而不是并行化的包装器。
    model_to_save = model.module if hasattr(model, 'module') else model
    # 定义模型的配置和权重的文件名。
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    # 定义模型保存的目录路径。
    output_dir = os.path.join(args.output_dir, sub_folder)
    # 如果模型保存的目录不存在，那么创建这个目录。
    os.makedirs(output_dir, exist_ok=True)
    # 定义模型权重和配置的保存路径。
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    # 获取模型的状态字典，这个字典包含了模型的所有权重。
    save_dict = model_to_save.state_dict()
    # 遍历状态字典中的所有键。这里使用list(save_dict.keys())而不是直接使用
    # save_dict.keys()是因为我们在遍历过程中会修改字典，直接遍历字典的键在Python中会导致错误。
    for key in list(save_dict.keys()):
        # if "lora" in key:：如果键中包含"lora"，那么删除这个键值对。
        # "lora"可能是某种特定的模型组件或参数，这里将其排除在保存的模型权重之外。
        if "lora" in key:
            del save_dict[key]
    # 将处理过的状态字典保存到文件中。
    torch.save(save_dict, output_model_file)
    # 将模型的配置保存为JSON文件。
    model_to_save.config.to_json_file(output_config_file)
    # 将分词器的词汇表保存到指定的目录。
    tokenizer.save_vocabulary(output_dir)

# 这段代码的目的是为所有可能用到的随机数生成器设定种子，以确保实验结果的可复现性。
# 随机数生成器的种子是生成随机数序列的初始值，当我们用同一个种子初始化随机数生成器时，它会生成完全一样的随机数序列。
def set_random_seed(seed):
    # 首先检查种子是否是None。如果是None，那么就跳过这段代码，随机数生成器将会用一个随机的种子。
    if seed is not None:
        set_seed(seed)# 这一行调用transformers库的set_seed的函数，将种子设定为指定的值。
        random.seed(seed) # 将Python内置的随机数生成器的种子设定为指定的值。
        np.random.seed(seed) # 将NumPy的随机数生成器的种子设定为指定的值。
        torch.manual_seed(seed) # 将PyTorch的随机数生成器的种子设定为指定的值。
        torch.cuda.manual_seed_all(seed) # 将PyTorch的所有GPU随机数生成器的种子设定为指定的值。

# 这段代码是在分布式训练环境中进行平均值计算的函数，通过这段代码，
# 所有的处理器（或者叫节点）上的同一个tensor都会被加和起来，然后除以总的处理器数，得到平均值。
def get_all_reduce_mean(tensor):
    # 这行代码执行一个分布式的reduce操作。reduce操作是指所有处理器中的同一个tensor都被某种方式结合起来。
    # 在这个例子中，torch.distributed.ReduceOp.SUM表示所有处理器上的tensor将被加和起来。
    # 加和的结果会在所有处理器上都可用。
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    # 这行代码将前一步得到的加和结果除以处理器的数量（也叫作 world size）。
    # 这样，tensor就变成了所有处理器上原始tensor的平均值。
    tensor = tensor / torch.distributed.get_world_size()
    # 最后，这个平均值tensor被返回。在所有处理器上，这个函数返回的tensor都是相同的，
    # 等于所有处理器上原始tensor的平均值。
    return tensor

# 打印 get_optimizer_grouped_parameters 的返回值
def debugOGP(optimizer_grouped_parameters):
    for id1, pg in enumerate(optimizer_grouped_parameters): #根据下面代码，这是固定3个！
        for id2, p in enumerate(pg["params"]):
            print_rank_0(f'T OGP:{id1}-params-{id2}:' + infoTensor(pg["params"][id2]))
        print(f'T OGP:{id1}-weight_decay:' + str(pg["weight_decay"]))
        if 'lr' in pg.keys():
            print_rank_0(f'T OGP:{id1}-lr:' + str(pg["lr"]))

# 这段代码的作用是将模型中的参数分组以便于在优化器中使用。它将模型参数分为两组：
# 一组需要进行权重衰减（L2正则化）的参数，另一组不需要进行权重衰减的参数。
def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    #yknote代码有变动
    # 它定义了一个列表 optimizer_grouped_parameters，其中包含两个字典。
    # 每个字典都对应一个参数组，包含 "params" 和 "weight_decay" 这两个关键字。
    optimizer_grouped_parameters = [
        # 在第一个字典中，它从模型参数中选出那些名称不包含 "bias" 或 "LayerNorm.weight"
        # 且需要求梯度的参数。这些参数在优化过程中会应用 weight_decay 作为权重衰减项。
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
		# 在第二个字典中，它选出那些名称包含 "bias" 或 "LayerNorm.weight" 且需要求梯度的参数。
        # 这些参数在优化过程中不会应用权重衰减，即其 "weight_decay" 值为0。
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters
    # 这种参数的分组策略是很常见的。比如在训练Transformer模型时，通常会为权重和偏置项设定不同的学习策略。
    # 这是因为权重衰减对于防止过拟合很有帮助，但对于某些参数（如偏置项或者层归一化的权重）可能会导致性能下降，因此常常会排除这些参数不进行权重衰减。


# 这个函数的主要功能是筛选出那些在DeepSpeed Zero 3优化中被离线存储，但在当前还未获取的参数。
# 在DeepSpeed Zero 3优化中，一些模型参数在使用过后会被离线存储，以此释放GPU显存。
# 当这些参数需要再次被使用时，需要先获取到本地。
def _z3_params_to_fetch(param_list):
    # 这个条件语句判断一个参数是否是被DeepSpeed Zero 3优化过的，且其状态为"未获取"（NOT_AVAILABLE）。
    # 对于被DeepSpeed Zero 3优化过的参数，它们有一个ds_id属性和一个ds_status属性，其中ds_status表示参数的当前状态。
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))

# 这个函数的主要作用是保存一个使用了DeepSpeed Zero优化（可能为stage 3）的模型。
# DeepSpeed的Zero优化技术是为了解决模型参数、优化器状态和梯度等内存占用问题，
# 通过这种方式，可以训练比当前GPU内存更大的模型。
def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    # 首先，检查输入的zero_stage是否为3，确定是否使用了DeepSpeed Zero阶段3优化。
    zero_stage_3 = (zero_stage == 3)
    # 然后，确保保存模型的目录存在。
    os.makedirs(save_dir, exist_ok=True)
    # 定义模型权重保存的完整路径。
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    # 如果模型是被包裹在其它结构（如DataParallel或DistributedDataParallel）中的，我们需要取出真实的模型实例。
    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        debuginfo(prj='ds-chat', info="Not use zero3")
      # 如果没有使用Zero阶段3优化，直接使用PyTorch的torch.save函数保存模型状态。
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        debuginfo(prj='ds-chat', info = "use zero3")
        # 如果使用了Zero阶段3优化，因为模型的部分参数和优化器状态在不同的设备上，所以需要先将它们收集起来。
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                # deepspeed.zero.GatheredParameters是DeepSpeed提供的一个上下文管理器，
                # 它可以将分布在多个设备上的参数收集到一起。这部分参数保存在CPU上。
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            # 然后，将收集好的参数（并且不包含“lora”关键字的参数）添加到输出状态字典中。
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        # 最后，再使用torch.save函数保存模型状态。
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)

        '''
        不同阶段输出不同，同一个ph同一个程序也有不同输出！！，仅仅在z3出现
        z123都可能是空的字典
        或者非常大的输出
        '''
        print("++++++++++++++++++content of output_state_dict ++++++++++++++++++++++++")
        if len(output_state_dict.keys()) != 0:
            for k in output_state_dict.keys():
                infoTen = infoTensor(output_state_dict[k])
                print(f"(### {k} is: {infoTen}")
        else:
            print("output_state_dict is", output_state_dict)
        print("++++++++++++++++++content of output_state_dict ++++++++++++++++++++++++")


        # 同时为了节省内存，使用del关键字删除了存储参数的字典。
        del output_state_dict


