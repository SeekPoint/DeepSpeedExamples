# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Run all steps with default settings:
$ python3 train.py

Change the model used for each step:
$ python3 train.py --actor-model 350m --reward-model 1.3b

Change the ZeRO stage used for actor/reward models:
$ python3 train.py --actor-zero-stage 1 --reward-zero-stage 3

Run a subset of the steps:
$ python3 train.py --step 1 2

Note: Step 3 relies on models trained in Steps 1 & 2. If you have already
trained these models, you can run just Step 3 and select which models from
Steps 1 & 2 to use. For example, let's train models for Steps 1 & 2 using
125m and 350m models:
$ python3 train.py --step 1 2 --actor-model 125m --reward-model 125m
$ python3 train.py --step 1 2 --actor-model 350m --reward-model 350m

Now we can run Step 3 with any combination of these models:
$ python3 train.py --step 3 --actor-model 125m --reward-model 350m
$ python3 train.py --step 3 --actor-model 350m --reward-model 125m
"""

import argparse
import warnings
import subprocess # 用于创建新的进程
import os
import datetime
import time

from pydebug import debuginfo

# 定义了三个训练步骤的目录
step_dirs = {
    1: "training/step1_supervised_finetuning",
    2: "training/step2_reward_model_finetuning",
    3: "training/step3_rlhf_finetuning",
}
model_type = {1: "actor", 2: "reward", 3: "step3"}
dse_url = "https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat/"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        type=int,
        nargs="+", # 可以接受一个或多个参数
        choices=(1, 2, 3), # 限制输入的值只能是1, 2, 3中的一个
        default=(1, 2, 3),
        help="Which steps of the ChatGPT example to run",
    )
    parser.add_argument(
        "--actor-model",
        # actor模型只能选用“facebook/opt”系列的1.3b~66b  ???yknote
        # 但即使是1.3b也仍对设备显存有一定的要求
        # 对于使用较弱设备进行学习的用户不是很友好
        type=lambda x: x.replace("facebook/opt-", ""),
        default="125m",
        choices=("125m", "350m", "1.3b", "6.7b", "13b", "66b"),
        help="Which facebook/opt-* model to use for Actor (step 1)",
    )
    parser.add_argument(
        "--reward-model",
        type=lambda x: x.replace("facebook/opt-", ""),
        default="125m",
        choices=("125m", "350m", "1.3b", "6.7b", "13b", "66b"),
        help="Which facebook/opt-* model to use for Reward (step 2)",
    )
    parser.add_argument(
        "--actor-zero-stage", # 指定在第一步（Actor）训练中使用的ZeRO优化的阶段
        type=str,
        default="",
        choices=("", "0", "1", "2", "3"),
        help="ZeRO stage for step 1 (Actor) training",
    )
    parser.add_argument(
        "--reward-zero-stage", # 指定在第二步（Critic）训练中使用的ZeRO优化的阶段
        type=str,
        default="",
        choices=("", "0", "1", "2", "3"),
        help="ZeRO stage for step 2 (Critic) training",
    )
    parser.add_argument(
        "--output-dir",
        type=lambda x: os.path.abspath(x),
        default="./output",
        help="Directory for output of each step",
    )
    parser.add_argument(
        "--deployment-type",
        type=str,
        default="single_gpu",
        choices=("single_gpu", "single_node", "multi_node"),
        help="Number of GPUs to run the actor/reward models on",
    )
    args = parser.parse_args()

    if args.actor_zero_stage != "" or args.reward_zero_stage != "":
        # 择非默认的优化级别可能会导致Out-Of-Memory（OOM）错误，或者导致性能降低
        warnings.warn(
            "Non-default zero stages may result in OOM errors or worse performance."
        )

    return args


def get_model_size(args, step_num):
    '''获取模型大小'''
    if step_num == 3:
        return get_model_size(args, 1)

    # model_type是一个字典，其键值分别为步骤号和对应的模型类型
    return getattr(args, f"{model_type[step_num]}_model")


def get_zero_stage(args, step_num):
    '''获取ZeRO阶段'''
    return getattr(args, f"{model_type[step_num]}_zero_stage")


def get_output_dir(args, step_num):
    '''输出路径'''
    model_size = get_model_size(args, step_num)
    output_dir = os.path.join(args.output_dir,
                              f"{model_type[step_num]}-models",
                              f"{model_size}")
    return output_dir


def get_script(args, step_num):
    '''根据step_num和命令行参数（args）来获取需要执行的脚本文件的路径'''
    model_size = get_model_size(args, step_num)

    script = os.path.join(
        os.getcwd(), # 当前工作目录
        step_dirs[step_num], # 步骤目录
        "training_scripts",
        args.deployment_type, # 部署类型
        f"run_{model_size}.sh", # 模型的脚本文件名
    )

    # 检查这个脚本文件是否真实存在
    assert os.path.isfile(
        script
    ), f"{script} does not exist.\n\n Use examples in {os.path.dirname(script)} as a template."

    return script


def verify_model(args, step_num):
    '''验证给定步骤的模型是否已经被训练过'''
    # 获取模型输出的目录
    output_dir = get_output_dir(args, step_num)

    # 获取模型的大小
    model_size = get_model_size(args, step_num)

    # 模型文件的路径
    model_file = os.path.join(output_dir, "pytorch_model.bin")

    # 检查模型文件是否存在
    if not os.path.isfile(model_file):
        # 创建一个错误信息
        error_str = f"Step {step_num} model has not been trained. Train it with:\n"
        error_str += f"python3 train.py --step {step_num}"
        error_str += f" --{model_type[step_num]}-model {model_size}"
        raise RuntimeError(error_str)


def get_cmd(args, step_num):
    '''生成执行每个训练步骤的命令行命令'''
    # 获取模型的输出目录
    output_dir = get_output_dir(args, step_num)

    # 获取用于执行训练的bash脚本的路径
    script = get_script(args, step_num)
    gd.debuginfo(prj="ds_chat", info=f"output_dir is:", output_dir)
    gd.debuginfo(prj="ds_chat", info=f"script is:", script)

    if step_num in (1, 2):
        # 获取到对应的ZeRO阶段
        zero_stage = get_zero_stage(args, step_num)

        # 包含了脚本路径，输出目录，以及ZeRO阶段
        cmd = f"bash {script} {output_dir} {zero_stage}"

    if step_num == 3:
        # 验证第一步和第二步的模型是否存在
        verify_model(args, 1)  # Verify step 1 model exists
        verify_model(args, 2)  # Verify step 2 model exists

        # 获取第一步和第二步的输出目录和ZeRO阶段
        s1_dir, s1_zs = get_output_dir(args, 1), get_zero_stage(args, 1)
        s2_dir, s2_zs = get_output_dir(args, 2), get_zero_stage(args, 2)
        cmd = f"bash {script} {s1_dir} {s2_dir} '{s1_zs}' '{s2_zs}' {output_dir}"

    return cmd


def launch_cmd(args, step_num, cmd):
    '''在指定的工作目录中执行命令cmd'''
    # 获取当前步骤的工作目录
    working_dir = step_dirs[step_num]
    print(f"Running:\n{cmd}")

    # 启动一个子进程来执行这个命令
    # 这里的cwd参数设置为working_dir，表示命令在working_dir目录下执行。
    # shell=True表示在一个shell中执行这个命令。
    p = subprocess.Popen(cmd, cwd=working_dir, shell=True)

    # 等待子进程结束
    p.wait()

    # 如果子进程返回的状态码不为0，表示执行过程中发生错误，
    # 那么就会抛出一个RuntimeError异常。
    if p.returncode != 0:
        raise RuntimeError('\n\n'.join((
            f"Step {step_num} exited with non-zero status {p.returncode}",
            f"Launch command: {cmd}",
            f"Log output: {os.path.join(get_output_dir(args, step_num), 'training.log')}",
            f"Please see our tutorial at {dse_url}{step_dirs[step_num]}",
            "Please check that you have installed our requirements: `pip install -r requirements.txt`",
            f"If you are seeing an OOM error, try modifying {get_script(args, step_num)}:",
            "  - Reduce `--per_device_*_batch_size`",
            "  - Increase `--zero_stage {0,1,2,3}` on multi-gpu setups",
            "  - Enable `--gradient_checkpointing` or `--only_optimize_lora`"
        )))


def main(args):
    # 记录了开始时间
    start_time = time.time()
    for step_num in args.step:
        print(f"---=== Running Step {step_num} ===---")
        step_start_time = time.time()

        # 返回一个字符串，表示用于运行当前步骤的shell命令
        cmd = get_cmd(args, step_num)

        # 执行上面的命令
        launch_cmd(args, step_num, cmd)

        # 记录当前步骤的结束时间
        step_time = int(time.time() - start_time)

        # 计算出消耗的时间
        time_str = str(datetime.timedelta(seconds=step_time))
        print(f"---=== Finished Step {step_num} in {time_str} ===---")

    # 计算出所有步骤消耗的总时间
    total_time = int(time.time() - start_time)
    time_str = str(datetime.timedelta(seconds=total_time))

    if len(args.step) > 1:
        print(f"---=== Finished Steps {args.step} in {time_str} ===---")


if __name__ == "__main__":
    # self 不存在时，也可以保证程序正常运行！！https://blog.csdn.net/jewely/article/details/90259422
    gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ if 'self' in locals() or 'self' in globals() else '')
    args = parse_args()
    gd.debuginfo(prj="ds_chat", info=f"args is:", args)
    main(args)
