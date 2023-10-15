# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 训练好的模型所在的目录
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model")

    # 在每次响应中最多生成的新tokens的数量
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    args = parser.parse_args()

    # 构建一个Python子进程，运行chatbot.py脚本，将args.path和args.max_new_tokens作为参数传入。
    cmd = f"python3 ./inference/chatbot.py --path {args.path} --max_new_tokens {args.max_new_tokens}"

    # subprocess.Popen创建了一个子进程，参数shell=True表示使用系统的shell执行命令。
    p = subprocess.Popen(cmd, shell=True)

    # p.wait()则是等待子进程结束，这个函数会阻塞主进程，直到子进程结束。
    p.wait()
