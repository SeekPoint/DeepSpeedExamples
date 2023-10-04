# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import re
import logging
import transformers  # noqa: F401
import os
import json
from transformers import pipeline, set_seed
from transformers import AutoConfig, OPTForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser() # 这行代码创建了一个 ArgumentParser 对象，它将保存所有需要解析的命令行参数信息。
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model") # 这行代码添加了一个命令行选项 --path。这个选项的值是一个字符串，作为帮助文本，它显示为 "Directory containing trained actor model"。当用户在命令行中使用 --path 选项时，用户需要提供一个路径，这个路径是训练过的 actor 模型的目录。
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    ) # 这行代码添加了另一个命令行选项 --max_new_tokens。这个选项的值是一个整数，默认值为 128。作为帮助文本，它显示为 "Maximum new tokens to generate per response"。当用户在命令行中使用 --max_new_tokens 选项时，用户可以指定生成每个响应的最大新token数。
    args = parser.parse_args() # 这行代码调用了 parser 对象的 parse_args 方法，该方法将读取命令行参数，并将它们转化为一个命名空间，这个命名空间存储了每个参数的名称和对应的值。
    return args # 最后，函数返回这个命名空间，这样其他代码就可以通过这个命名空间来访问解析得到的命令行参数的值。

# 这段代码的主要目的是加载一个预训练的 transformer 模型，并返回一个文本生成器
def get_generator(path):
    if os.path.exists(path): # 检查 path 是否指向一个真实存在的文件路径。如果存在，进入下面的代码块。
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(path, "config.json") # 定义一个新的路径，这个路径是模型路径下的 config.json 文件，通常包含模型的配置信息。
        if os.path.exists(model_json): # 检查 config.json 文件是否存在。
            model_json_file = json.load(open(model_json)) # 打开并加载 config.json 文件的内容。
            model_name = model_json_file["_name_or_path"] # 从 json 文件中获取模型的名字或路径。
            tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                      fast_tokenizer=True) # 使用模型名字或路径来加载预训练的 tokenizer。这里使用的是 AutoTokenizer 类，它会根据模型的类型自动选择合适的 tokenizer。fast_tokenizer=True 会尽可能使用 Hugging Face 的快速 tokenizer。
    else:# 如果 path 并非一个实际存在的文件路径，执行下面的代码。
        tokenizer = AutoTokenizer.from_pretrained(path, fast_tokenizer=True) #直接使用 path 作为预训练模型的名字或路径来加载 tokenizer。

    tokenizer.pad_token = tokenizer.eos_token # 将 tokenizer 的 pad_token 设为和 eos_token 一样的 token。这可能是因为在文本生成中，模型可能需要对输入进行 padding。

    model_config = AutoConfig.from_pretrained(path) # 从预训练模型路径或名字加载模型的配置。
    model = OPTForCausalLM.from_pretrained(path,
                                           from_tf=bool(".ckpt" in path),
                                           config=model_config).half() #从预训练模型路径或名字加载模型。这里的模型类型是 OPTForCausalLM，它适用于生成文本的任务。from_tf=bool(".ckpt" in path) 会检查模型是否从 TensorFlow checkpoint 加载，如果路径中包含 ".ckpt"，则返回 True。.half() 将模型转为半精度模型，可以减少内存占用和提高运算速度。

    model.config.end_token_id = tokenizer.eos_token_id # 设置模型配置的 end_token_id 为 tokenizer 的 eos_token_id。
    model.config.pad_token_id = model.config.eos_token_id # 设置模型配置的 pad_token_id 为 eos_token_id。
    model.resize_token_embeddings(len(tokenizer)) # 重新设置模型的 token embeddings 的大小以适应 tokenizer 的大小。
    generator = pipeline("text-generation",
                         model=model,
                         tokenizer=tokenizer,
                         device="cuda:0") # 使用模型和 tokenizer 创建一个文本生成 pipeline，设备设置为 "cuda:0"。
    return generator # 返回这个文本生成器。

def get_user_input(user_input):
    tmp = input("Enter input (type 'quit' to exit, 'clear' to clean memory): ") # 这行代码使用 Python 的内置 input 函数提示用户输入内容。用户输入的内容会被赋值给 tmp 变量。
    new_inputs = f"Human: {tmp}\n Assistant: " # f"Human: {tmp}\n Assistant: "：这行代码创建一个新的字符串 new_inputs，其中包含 "Human: "，然后是用户输入的内容，然后是一个换行符，最后是 "Assistant: "。这个地方是opt模型的prompt工程技巧，如果是其它模型也需要对应修改。
    user_input += f" {new_inputs}" # 这行代码将 new_inputs 添加到 user_input 字符串的末尾，user_input 可能是用来记录整个对话过程。
    return user_input, tmp == "quit", tmp == "clear" # 最后，函数返回三个值：更新过的 user_input 字符串;一个布尔值，当用户输入 "quit" 时，这个值为 True，否则为 False。这可能是一个信号，告诉主程序用户是否想要退出;另一个布尔值，当用户输入 "clear" 时，这个值为 True，否则为 False。这可能是一个信号，告诉主程序用户是否想要清除内存。


def get_model_response(generator, user_input, max_new_tokens):
    response = generator(user_input, max_new_tokens=max_new_tokens) # 这行代码调用 generator 函数（或模型），并将 user_input 和 max_new_tokens 作为参数传递。该行代码生成一个响应，也就是生成器根据 user_input 生成的文本，长度不超过 max_new_tokens。
    return response # 然后，函数返回这个生成的响应。


def process_response(response, num_rounds):
    output = str(response[0]["generated_text"]) # 这行代码从响应中取出生成的文本，并将其转换为字符串。
    output = output.replace("<|endoftext|></s>", "") # 这行代码将输出中的所有<|endoftext|></s>标签（如果有的话）替换为空字符串。感觉这里代码有点问题，应该是把<|endoftext|><和</s>分别替换为空字符串才对？上面serving演示截图里面也可以看出这个问题。
    all_positions = [m.start() for m in re.finditer("Human: ", output)]# 这行代码使用正则表达式查找输出中所有"Human: "字符串的开始位置，并将这些位置存储在all_positions列表中。
    place_of_second_q = -1
    if len(all_positions) > num_rounds: # 然后，函数检查"Human: "字符串出现的次数是否大于已经进行的对话轮数。如果是，那么第num_rounds个"Human: "字符串的位置就是place_of_second_q。如果不是，那么place_of_second_q将保持为-1。
        place_of_second_q = all_positions[num_rounds]
    if place_of_second_q != -1: # 如果place_of_second_q不等于-1（也就是说，输出中有超过num_rounds个"Human: "字符串），那么输出将被截取到第num_rounds个"Human: "字符串的位置。否则，输出将保持不变。
        output = output[0:place_of_second_q]
    return output

def main(args):
    generator = get_generator(args.path) # 调用 get_generator 函数，根据在 args.path 中指定的路径获取生成器模型。
    set_seed(42) # 设置随机数生成器的种子为 42，以保证结果的可重复性。

    user_input = ""
    num_rounds = 0 # 初始化用户输入为空字符串，并设置对话轮数为 0。
    while True: # 这是一个无限循环，它将持续进行直到用户输入“quit”。
        num_rounds += 1
        user_input, quit, clear = get_user_input(user_input) # 在循环的每一轮中，首先增加 num_rounds 的值，然后调用 get_user_input 函数获取用户的输入。

        if quit:
            break
        if clear:
            user_input, num_rounds = "", 0
            continue # 如果用户输入“quit”，则跳出循环。如果用户输入“clear”，则清空用户输入和对话轮数，并开始下一轮循环。

        response = get_model_response(generator, user_input,
                                      args.max_new_tokens) # 然后，调用 get_model_response 函数，传入生成器、用户输入和 args.max_new_tokens，获取模型的响应。
        output = process_response(response, num_rounds) # 之后，使用 process_response 函数处理模型的响应，并将结果输出到控制台。

        print("-" * 30 + f" Round {num_rounds} " + "-" * 30)
        print(f"{output}")
        user_input = f"{output}\n\n" # 最后，更新 user_input 为当前的输出，然后进入下一轮循环。


if __name__ == "__main__":
    # Silence warnings about `max_new_tokens` and `max_length` being set
    logging.getLogger("transformers").setLevel(logging.ERROR)

    args = parse_args()
    # 将 transformers 的日志级别设置为 ERROR，以消除关于 max_new_tokens 和 max_length 设置的警告。
    # 然后，调用 parse_args 函数解析命令行参数，并将结果传递给 main 函数。
    main(args)

# Example:
"""
 Human: what is internet explorer?
 Assistant:
Internet Explorer is an internet browser developed by Microsoft. It is primarily used for browsing the web, but can also be used to run some applications. Internet Explorer is often considered the best and most popular internet browser currently available, though there are many other options available.

 Human: what is edge?
 Assistant:
 Edge is a newer version of the Microsoft internet browser, developed by Microsoft. It is focused on improving performance and security, and offers a more modern user interface. Edge is currently the most popular internet browser on the market, and is also used heavily by Microsoft employees.
"""
