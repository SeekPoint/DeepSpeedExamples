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
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from pydebug import gd, infoTensor

# 在rank0也就是master rank打印信息，防止每个机器或GPU都打印消息造成大量重复信息
def print_rank_0(msg, rank=0):
    '''用于在多机或多进程分布式训练中，控制只有特定的进程(rank=0)才打印信息。
    这是为了防止在并行环境中，每个进程都打印相同的信息，从而导致日志的冗余。'''
    if rank <= 0:
        print(msg)

# 这个函数的作用是把一个数据批次移动到指定的设备上。在PyTorch中，数据默认是在CPU上的，
# 如果要在GPU上进行运算，就需要先把数据移到GPU上。这个函数通过遍历批次中的所有元素并
# 调用to(device)方法来实现这一点。如果某个元素不能被移到指定的设备上
#（例如，这个元素是一个字符串或者其他不能在GPU上使用的类型），那么就直接保留这个元素，不进行任何操作。
def to_device(batch, device):
    '''将输入的批次数据（batch）移到指定的设备（device）上'''
    # 保存处理后的batch数据
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


# 这段代码定义了一个名为load_hf_tokenizer的函数，该函数的作用是加载一个指定的分词器。
# 这个函数的参数有：
# model_name_or_path：模型的名称或者预训练模型的路径。
# fast_tokenizer：是否使用Fast Tokenizer，这是一种更快的分词器实现。
def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    # 检查model_name_or_path是否存在。如果存在，可能意味着这是一个本地的模型路径。
	# 如果给定的模型路径或名称是一个存在的本地路径，函数会从该路径加载tokenizer
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        # 如果是本地模型，再检查在该路径下的"config.json"文件是否存在。
        # 如果存在，打开该文件，并从文件中读取"_name_or_path"字段，将其赋值给model_name，
        # 然后通过AutoTokenizer.from_pretrained函数，使用该模型名称和fast_tokenizer参数，从HuggingFace模型库下载并加载分词器。
		# 检查在该路径下是否有"config.json"文件，这个文件通常包含了用于初始化模型或tokenizer的配置信息。
        model_json = os.path.join(model_name_or_path, "config.json")

        if os.path.exists(model_json):
            model_json_file = json.load(open(model_json))
            model_name = model_json_file["_name_or_path"]
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, fast_tokenizer=fast_tokenizer)
    else:
        # 如果model_name_or_path不存在，直接使用AutoTokenizer.from_pretrained函数，
        # 使用model_name_or_path和fast_tokenizer参数，从HuggingFace模型库下载并加载分词器。
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
    return tokenizer

# 将模型和 tokenizer 以 Hugging Face 的格式保存
# 这段代码的功能是将训练好的模型以Hugging Face格式保存，这样之后就可以使用Hugging Face库的from_pretrained方法加载了。
def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    # 首先，这行代码检查模型是否有'module'这个属性。这在PyTorch中是一个常见的模式，
    # 当模型被封装在torch.nn.DataParallel或torch.nn.parallel.DistributedDataParallel时，
    # 模型的所有属性都会被存储在'module'属性中。
    # 所以这行代码的目的是确保我们总是在原始模型上进行操作，而不是并行化的包装器。
	# 检查模型是否使用了torch.nn.DataParallel或者torch.nn.parallel.DistributedDataParallel。
    # 如果使用了，模型会被包装在一个名为module的属性中。
    model_to_save = model.module if hasattr(model, 'module') else model

    # 定义模型的配置和权重的文件名。
	
	# 模型配置文件
    CONFIG_NAME = "config.json"

    # 模型权重的文件名
    WEIGHTS_NAME = "pytorch_model.bin"

    # 定义模型保存的目录路径。 # 创建输出目录
    output_dir = os.path.join(args.output_dir, sub_folder)

    # 如果模型保存的目录不存在，那么创建这个目录。
    os.makedirs(output_dir, exist_ok=True)

    # 定义模型权重和配置的保存路径。
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)

    # 获取模型的状态字典，这个字典包含了模型的所有权重。 # 模型的状态字
    save_dict = model_to_save.state_dict()

    # 遍历状态字典中的所有键。这里使用list(save_dict.keys())而不是直接使用
    # save_dict.keys()是因为我们在遍历过程中会修改字典，直接遍历字典的键在Python中会导致错误。
	# 遍历字典的所有键，并删除其中包含lora的键
    for key in list(save_dict.keys()):
        # if "lora" in key:：如果键中包含"lora"，那么删除这个键值对。
        # "lora"可能是某种特定的模型组件或参数，这里将其排除在保存的模型权重之外。
        if "lora" in key:
            del save_dict[key]

    # 将处理过的状态字典保存到文件中。 # 保存模型的配置
    torch.save(save_dict, output_model_file)

    # 将模型的配置保存为JSON文件。
    model_to_save.config.to_json_file(output_config_file)

    # 将分词器的词汇表保存到指定的目录。 # 保存tokenizer的词汇表
    tokenizer.save_vocabulary(output_dir)

# 这段代码的目的是为所有可能用到的随机数生成器设定种子，以确保实验结果的可复现性。
# 随机数生成器的种子是生成随机数序列的初始值，当我们用同一个种子初始化随机数生成器时，它会生成完全一样的随机数序列。
def set_random_seed(seed):
    # 首先检查种子是否是None。如果是None，那么就跳过这段代码，随机数生成器将会用一个随机的种子。
    '''设置不同库的随机数生成器的种子，确保在给定相同输入的情况下，
       代码生成的所有随机数在不同的运行中都是相同的。'''
    if seed is not None:
        # 这一行调用transformers库的set_seed的函数，将种子设定为指定的值。
		# 设置了Hugging Face库自己的随机数生成器的种子
        set_seed(seed)

        # 将Python内置的随机数生成器的种子设定为指定的值。
		# 设置了Python内置random模块的随机数生成器的种子
        random.seed(seed)

        # 将NumPy的随机数生成器的种子设定为指定的值。
		# 设置了NumPy的随机数生成器的种子
        np.random.seed(seed)

        # 将PyTorch的随机数生成器的种子设定为指定的值。
		# 设置了PyTorch的随机数生成器的种子
        torch.manual_seed(seed)

        # 将PyTorch的所有GPU随机数生成器的种子设定为指定的值。
		# 设置了PyTorch基于CUDA的随机数生成器的种子
        torch.cuda.manual_seed_all(seed)

# 这段代码是在分布式训练环境中进行平均值计算的函数，通过这段代码，
# 所有的处理器（或者叫节点）上的同一个tensor都会被加和起来，然后除以总的处理器数，得到平均值。
def get_all_reduce_mean(tensor):
    '''用于实现分布式环境中的全局平均值计算'''
    # 实现了所有分布式进程中的tensor的值进行reduce操作，tensor的值在所有进程中进行了相加，得到一个总和。
	
    # 这行代码执行一个分布式的reduce操作。reduce操作是指所有处理器中的同一个tensor都被某种方式结合起来。
    # 在这个例子中，torch.distributed.ReduceOp.SUM表示所有处理器上的tensor将被加和起来。
    # 加和的结果会在所有处理器上都可用。
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    
    # 这行代码将前一步得到的加和结果除以处理器的数量（也叫作 world size）。
    # 这样，tensor就变成了所有处理器上原始tensor的平均值。
    # get_world_size() : 返回的是分布式环境中的总进程数
    # tensor : 所有进程的平均值
    tensor = tensor / torch.distributed.get_world_size()

    # 最后，这个平均值tensor被返回。在所有处理器上，这个函数返回的tensor都是相同的，
    # 等于所有处理器上原始tensor的平均值。
    return tensor

# 打印 get_optimizer_grouped_parameters 的返回值
def debugOGP(optimizer_grouped_parameters):
    for id1, pg in enumerate(optimizer_grouped_parameters): #根据下面代码，这是固定3个！
        for id2, p in enumerate(pg["params"]):
            print_rank_0(f'T OGP:{id1}-params-{id2}:' + infoTensor(pg["params"][id2]))
        gd.debuginfo(prj="ds_chat", info=f'T OGP:{id1}-weight_decay:' + str(pg["weight_decay"]))
        if 'lr' in pg.keys():
            print_rank_0(f'T OGP:{id1}-lr:' + str(pg["lr"]))

# 这段代码的作用是将模型中的参数分组以便于在优化器中使用。它将模型参数分为两组：
# 一组需要进行权重衰减（L2正则化）的参数，另一组不需要进行权重衰减的参数。
def get_optimizer_grouped_parameters(
    model, # 模型
    weight_decay, # 权重衰减的系数
    lora_lr=5e-4,
    no_decay_name_list=["bias", "LayerNorm.weight"],  # 不应用权重衰减的参数名字列表
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    # yknote代码有变动
    # 它定义了一个列表 optimizer_grouped_parameters，其中包含两个字典。
    # 每个字典都对应一个参数组，包含 "params" 和 "weight_decay" 这两个关键字。
    optimizer_grouped_parameters = [
        # 在第一个字典中，它从模型参数中选出那些名称不包含 "bias" 或 "LayerNorm.weight"
        # 且需要求梯度的参数。这些参数在优化过程中会应用 weight_decay 作为权重衰减项。
        {
            # 为了获取需要应用权重衰减的参数，首先遍历模型的所有参数，
            # 然后通过检查参数名称是否包含在no_decay_name_list列表中来决定是否应用权重衰减。
            # 如果参数名称不在no_decay_name_list列表中并且参数需要计算梯度，那么就将其加入到这一组。
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            # 权重衰减值
            "weight_decay": weight_decay,
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
		    # 为了获取不需要应用权重衰减的参数，同样遍历模型的所有参数，
            # 如果参数名称包含在no_decay_name_list列表中并且参数需要计算梯度，那么就将其加入到这一组。
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            # 权重衰减值设为0
            "weight_decay":
            0.0,
        },
    ]

    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)

    # 返回包含了两组参数的列表
    return optimizer_grouped_parameters
    # 这种参数的分组策略是很常见的。比如在训练Transformer模型时，通常会为权重和偏置项设定不同的学习策略。
    # 这是因为权重衰减对于防止过拟合很有帮助，
    # 但对于某些参数（如偏置项或者层归一化的权重）可能会导致性能下降，因此常常会排除这些参数不进行权重衰减。

# 这个函数的主要功能是筛选出那些在DeepSpeed Zero 3优化中被离线存储，但在当前还未获取的参数。
# 在DeepSpeed Zero 3优化中，一些模型参数在使用过后会被离线存储，以此释放GPU显存。
# 当这些参数需要再次被使用时，需要先获取到本地。
def _z3_params_to_fetch(param_list):
    # 这个条件语句判断一个参数是否是被DeepSpeed Zero 3优化过的，且其状态为"未获取"（NOT_AVAILABLE）。
    # 对于被DeepSpeed Zero 3优化过的参数，它们有一个ds_id属性和一个ds_status属性，其中ds_status表示参数的当前状态。

    # 对于列表param_list中的每个参数p，检查参数是否有ds_id属性。ds_id属性表明参数是分布在多个GPU上。
    # 检查参数的状态p.ds_status是否等于ZeroParamStatus.NOT_AVAILABLE。
    # 这个状态表明参数当前不在该GPU上，需要从其他GPU收集
	return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, # 原模型
                   model_ema, # 用于存储平均参数的模型
                   beta=0.992, # 滑动平均因子
                   device=None,
                   zero_stage=0 # DeepSpeed的ZeRO优化阶段
                   ):
    '''在训练深度学习模型时，将模型的参数更新为它们的移动平均值。
       这是训练过程中的常见技术，可以帮助稳定模型的训练并提高最终性能。'''

    # 是否在使用DeepSpeed的ZeRO-3阶段，
    # ZeRO-3是一种内存优化策略，用于分布式训练，它会将模型参数、优化器状态、和梯度分布在多个GPU上。
    zero_stage_3 = (zero_stage == 3)
    gd.debuginfo(prj="ds_chat", info=f"zero_stage_3={zero_stage_3}")
    with torch.no_grad():
        # 遍历模型的每个参数及其对应的滑动平均参数
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            gd.debuginfo(prj="ds_chat", info=f'param={param} #### param_ema={param_ema}')
            # TODO: use prefiltering for efficiency
            # 如果使用ZeRO-3阶段，找出列表中需要从其他GPU收集的参数。否则，返回空列表。
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            gd.debuginfo(prj="ds_chat", info=f"params_to_fetch={params_to_fetch}")

            # 是否需要在多个设备之间同步参数
            should_gather_param = len(params_to_fetch) > 0

            # 如果需要同步参数，就使用DeepSpeed的GatheredParameters上下文管理器来同步。
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):

                # 获取当前参数的数据
                data = param.data
                if device is not None:
                    data = data.to(device)

                # 使用PyTorch的lerp函数，根据权重beta在当前参数和平均参数之间做线性插值，然后将结果复制到平均参数中。
                # ① torch.lerp(data, param_ema.data, beta) : torch.lerp函数执行了线性插值，
                #    它接受三个参数，分别是start、end和weight。
                #    它的计算方式是start + weight * (end - start)，这个计算将得到一个新的值，
                #    这个值是data和param_ema.data之间的线性插值。
                #    beta这个权重决定了更多的侧重点放在哪里，beta越接近1，侧重点就越靠近param_ema.data，即平均值会更加关注过去的参数，
                #    beta越接近0，侧重点就越靠近data，即新的参数对平均值的影响会越大。
                # ② param_ema.data.copy_()：这部分是将上述计算结果（线性插值后的新值）复制到param_ema.data中。
                #    copy_是一个原地操作，会改变param_ema.data的值，但不会改变它的id或内存位置。
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))

# 在使用DeepSpeed的ZeRO Stage 3优化时，保存模型
# 这个函数的主要作用是保存一个使用了DeepSpeed Zero优化（可能为stage 3）的模型。
# DeepSpeed的Zero优化技术是为了解决模型参数、优化器状态和梯度等内存占用问题，
# 通过这种方式，可以训练比当前GPU内存更大的模型。
def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    '''在ZeRO Stage 3中，模型的参数、优化器状态和梯度被分布在不同的GPU中，因此保存模型需要特殊的操作。'''
    # 首先，检查输入的zero_stage是否为3，确定是否使用了DeepSpeed Zero阶段3优化。
    zero_stage_3 = (zero_stage == 3)

    # 然后，确保保存模型的目录存在。
    os.makedirs(save_dir, exist_ok=True)

    # 定义模型权重保存的完整路径。
    WEIGHTS_NAME = "pytorch_model.bin"

    # 模型文件的完整路径
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    # 如果模型使用了DataParallel或DistributedDataParallel，我们需要从module属性中获取模型。
    # 如果模型是被包裹在其它结构（如DataParallel或DistributedDataParallel）中的，我们需要取出真实的模型实例。
    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema

    # 如果没有使用ZeRO Stage 3，并且当前是主节点global_rank==0，直接保存模型的状态字典state_dict
    if not zero_stage_3:
        gd.debuginfo(prj="ds_chat", info=f"Not use zero3")

      # 如果没有使用Zero阶段3优化，直接使用PyTorch的torch.save函数保存模型状态。
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        gd.debuginfo(prj="ds_chat", info=f"use zero3")

        # 如果使用了Zero阶段3优化，因为模型的部分参数和优化器状态在不同的设备上，所以需要先将它们收集起来。
        output_state_dict = {}

        # 遍历模型的所有参数
        for k, v in model_to_save.named_parameters():
            gd.debuginfo(prj="ds_chat", info=f"k={k}")
            # save_zero_three_model k is model.decoder.layers.5.self_attn.q_proj.weight

            # gd.debuginfo(prj="ds_chat", info=f"v={v}")
            gd.debuginfo(prj="ds_chat", info=f"T: v={infoTensor(v)}")

            # 如果参数在分布式环境中（即 v.ds_id 存在）
            if hasattr(v, 'ds_id'):
                # 从各个 GPU 收集参数值
                # deepspeed.zero.GatheredParameters是DeepSpeed提供的一个上下文管理器，
                # 它可以将分布在多个设备上的参数收集到一起。这部分参数保存在CPU上。

                # gd.debuginfo(prj="ds_chat", info=f"[v]={[v]}")
                # [v]=[Parameter containing:
                # tensor([], device='cuda:1', dtype=torch.float16, requires_grad=True)]

                gd.debuginfo(prj="ds_chat", info=f"T: v={infoTensor(v)}")

                tmpz3 = _z3_params_to_fetch([v])
                gd.debuginfo(prj="ds_chat", info=f"tmpz3={tmpz3}")
                # tmpz3=[Parameter containing:
                # tensor([], device='cuda:0', dtype=torch.float16, requires_grad=True)]


                with deepspeed.zero.GatheredParameters(tmpz3,
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
                    # gd.debuginfo(prj="ds_chat", info=f"v_p---1={v_p}")
                    gd.debuginfo(prj="ds_chat", info=f"T: v_p---1 is + {infoTensor(v_p)}")
            else:
                # 直接获取参数值
                v_p = v.cpu()
                gd.debuginfo(prj="ds_chat", info=f"T: v_p---2={infoTensor(v_p)}")

            # 在主节点上，如果参数名称中不包含lora，将参数值添加到output_state_dict中。
            # 然后，将收集好的参数（并且不包含“lora”关键字的参数）添加到输出状态字典中。
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
                
        # 最后，再使用torch.save函数保存模型状态。
		# 在主节点上，将output_state_dict保存到文件中，并在保存后删除output_state_dict，以节省内存。
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)

        '''
        不同阶段输出不同，同一个ph同一个程序也有不同输出！！，仅仅在z3出现
        z123都可能是空的字典
        或者非常大的输出
        '''
        gd.debuginfo(prj="ds_chat", info=f"+++++++content of output_state_dict +++++++++++++++++")
        if len(output_state_dict.keys()) != 0:
            for k in output_state_dict.keys():
                infoTen = infoTensor(output_state_dict[k])
                gd.debuginfo(prj="ds_chat", info=f"(### {k}={infoTen}")
        else:
            gd.debuginfo(prj="ds_chat", info=f"output_state_dict={output_state_dict}")
        gd.debuginfo(prj="ds_chat", info=f"++++++++++++++++++content of output_state_dict ++++++++++++++++++++++++")


        # 同时为了节省内存，使用del关键字删除了存储参数的字典。
        del output_state_dict

def mem_estimate_log(args, exstr, model, num_gpus_per_node=2, num_nodes=1):
    logf = f'estimate_zeroX_model_states_mem_needs_all_live' + exstr
    if args is not None:
        if args.local_rank == 0:
            logf += f'_z{args.zero_stage}'
            gd.enable(info=logf)

        gd.debuginfo(prj='ds_chat', info=f"args.zero_stage={args.zero_stage}")

        if args.zero_stage == 2:
            estimate_zero2_model_states_mem_needs_all_live(model,
                                                           num_gpus_per_node=num_gpus_per_node, num_nodes=num_nodes)
        if args.zero_stage == 3:
            estimate_zero3_model_states_mem_needs_all_live(model,
                                                           num_gpus_per_node=num_gpus_per_node, num_nodes=num_nodes)

        if args.local_rank == 0:
            gd.disable(info=logf)
    else:
        gd.enable(info=logf)
        estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=num_gpus_per_node, num_nodes=num_nodes)
        estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=num_gpus_per_node, num_nodes=num_nodes)
        gd.disable(info=logf)

def mem_estimate_log_v2(args, exstr, model, num_gpus_per_node=2, num_nodes=1):
    logf = f'estimate_zeroX_model_states_mem_needs_all_live' + exstr
    if args is not None:
        logf += f'_actor_z{args.actor_zero_stage}_critic_z{args.critic_zero_stage}'
        if args.local_rank == 0:
            gd.enable(info=logf)

        gd.debuginfo(prj='ds_chat', info=f"args.actor_zero_stage={args.actor_zero_stage}")

        if args.actor_zero_stage == 2:
            estimate_zero2_model_states_mem_needs_all_live(model,
                                                           num_gpus_per_node=num_gpus_per_node, num_nodes=num_nodes)
        if args.actor_zero_stage == 3:
            estimate_zero3_model_states_mem_needs_all_live(model,
                                                           num_gpus_per_node=num_gpus_per_node, num_nodes=num_nodes)
        if args.local_rank == 0:
            gd.disable(info=logf)



'''
v_p---1=tensor([-0.1528,  0.1247, -0.0771,  0.0391,  0.0133,  0.0101, -0.1174, -0.1603,
        -0.1559,  0.1505, -0.0884,  0.1022, -0.0716,  0.1163, -0.0853,  0.1059],
       dtype=torch.float16)

save_zero_three_model v is tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.]], dtype=torch.float16)      
'''