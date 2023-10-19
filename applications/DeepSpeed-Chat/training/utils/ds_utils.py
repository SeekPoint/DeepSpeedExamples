# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4

# 这个函数的主要目标是生成一个字典，用来配置DeepSpeed训练模型的各种参数。
# DeepSpeed是一个用于加速大规模模型训练的库，其中包括一种叫做ZeRO（Zero Redundancy Optimizer）的优化技术，
# 可以大大减少模型训练过程中的显存和内存占用。
def get_train_ds_config(offload, # 否使用ZeRO Offload技术。如果为True，那么将模型参数和优化器状态offload到CPU，否则不使用offload。
                        stage=2,                    # 表示ZeRO的阶段。ZeRO有3个阶段（阶段0，阶段1，阶段2）
                        enable_hybrid_engine=False, # 是否启用混合引擎。混合引擎可以在训练时进行并行计算，从而提高训练效率。
                        inference_tp_size=1,        # 推理期间的流水线划分大小
                        release_inference_cache=False, # 是否在每次推理后释放缓存
                        pin_parameters=True,        # 是否固定参数。固定参数可以避免数据在CPU和GPU之间来回传输，从而提高训练效率。
                        tp_gather_partition_size=8, # 表示Gather操作的划分大小
                        max_out_tokens=512,         # 表示最大输出tokens
                        enable_tensorboard=False,
                        tb_path="",
                        tb_name=""):

    # 这行代码首先定义了一个变量device。如果参数offload为True，那么就将device设置为"cpu"，
    # 表示模型的参数和优化器状态将被"offload"到CPU内存中，以减少GPU显存的占用。
    # 如果offload为False，那么device就被设置为"none"，表示不进行offload。
    # 如果为True，模型参数和优化器状态将被卸载到CPU，这样可以节省GPU的显存使用量
    device = "cpu" if offload else "none"

    # 这行代码定义了一个字典zero_opt_dict，包含了ZeRO优化器的各种配置参数。
    # 设定DeepSpeed中的ZeRO(零冗余优化器)的相关参数
    zero_opt_dict = {
        "stage": stage, # ZeRO的阶段，可以为0、1、2、3，阶段越高，显存优化效果越好，但计算复杂度也越高。
        "offload_param": { # offload_param和offload_optimizer这两个字段控制是否将模型的参数和优化器状态offload到CPU内存。
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },

        # 只有在ZeRO stage 3时有效，控制参数offloading的细节。
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,

        # 这个字段控制是否使用显存更高效的线性层实现。
        # 由于当前的显存高效实现可能会导致数值不稳定，所以这里默认设为False。
        "memory_efficient_linear": False
    }

    # 返回一个配置字典，这个字典包含了训练模型的各种参数。
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE, # 全局的批次大小
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE, # 每个GPU的微批次大小。
        "steps_per_print": 10, # 每隔10步打印一次训练信息。
        "zero_optimization": zero_opt_dict, # 之前定义的ZeRO配置字典, ZeRO优化器
        "fp16": { # 控制是否使用混合精度训练，以及loss scaling的窗口大小。
            "enabled": True,           # 是否启用半精度浮点数进行训练
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,      # 梯度裁剪的阈值，防止梯度爆炸
        "prescale_gradients": False,   # 是否在梯度累计之前就进行梯度缩放，防止梯度下溢, 是否对梯度进行预缩放
        "wall_clock_breakdown": False, # 是否进行每步训练时间的详细分析, 是否打印训练过程中各部分的耗时详细信息
        "hybrid_engine": {  # HE允许模型的不同部分在不同的GPU上进行训练，以实现更大规模的模型训练和更高效的硬件利用率
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
