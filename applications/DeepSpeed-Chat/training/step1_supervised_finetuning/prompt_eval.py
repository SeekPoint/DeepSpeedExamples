# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import sys
import os

from transformers import (
    AutoModelForCausalLM, )

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_hf_model
from utils.utils import load_hf_tokenizer

logger = logging.getLogger(__name__)

# 这段代码定义了一个名为parse_args的函数，该函数的作用是解析命令行参数。
# 它使用Python的argparse库来完成这个工作。下面是每个参数的详解：
def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    # model_name_or_path_baseline：基线模型的路径，这是一个必须提供的参数（required=True）。
    parser.add_argument(
        "--model_name_or_path_baseline",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    # model_name_or_path_finetune：微调后模型的路径，这也是一个必须提供的参数。
    parser.add_argument(
        "--model_name_or_path_finetune",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    # num_beams：用于指定集束搜索的集束宽度，其默认值为1。
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    # num_beam_groups：用于指定集束搜索的组数，其默认值为1。
    parser.add_argument(
        "--num_beam_groups",
        type=int,
        default=1,
        help='Specify num of beams',
    )
    # top_k：用于指定在Top-K采样中考虑的最高可能性词汇的数量，其默认值为4。
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help='Specify num of beams',
    )
    # penalty_alpha：惩罚因子，其默认值为0.6。
    parser.add_argument(
        "--penalty_alpha",
        type=float,
        default=0.6,
        help='Specify num of beams',
    )
    # num_return_sequences：生成序列的数量，其默认值为1。
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help='Specify num of return sequences',
    )
    # max_new_tokens：生成的最大新token数，其默认值为100。
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help='Specify num of return sequences',
    )
    # language：语言类型，可以是"English"、"Chinese"或"Japanese"，默认为"English"。
    parser.add_argument("--language",
                        type=str,
                        default="English",
                        choices=["English", "Chinese", "Japanese"])
    # parser.parse_args()这个函数将解析命令行参数，并将结果保存在一个Namespace对象中。
    # 这个对象被返回，可以在其他地方使用这些参数。
    args = parser.parse_args()

    return args


# 这个函数是用来利用训练好的模型生成文本的，它接受以下参数
# model：已经训练好的模型。
# tokenizer：用于将文本转换为模型可理解的输入的工具。
# inputs：模型的输入数据。
# num_beams：在使用束搜索算法时的束宽，其默认值为1。
# num_beam_groups：在使用分组束搜索时的组数，默认为1。
# do_sample：是否进行随机采样。如果设为True，则在生成过程中会随机选择下一个单词，而不是仅选择最可能的单词。默认为False。
# num_return_sequences：模型返回的序列数，默认为1。
# max_new_tokens：模型生成的最大新token数，即最大生成文本的长度，默认为100。
def generate(model,
             tokenizer,
             inputs,
             num_beams=1,
             num_beam_groups=1,
             do_sample=False,
             num_return_sequences=1,
             max_new_tokens=100):
    # 函数首先使用模型的generate方法，根据提供的参数生成文本。
    generate_ids = model.generate(inputs.input_ids,
                                  num_beams=num_beams,
                                  num_beam_groups=num_beam_groups,
                                  do_sample=do_sample,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)
    # 使用tokenizer的batch_decode方法将生成的令牌ID解码为可读的文本。注意，这里跳过了特殊的令牌（如填充和开始/结束令牌），并且不会清理tokenize产生的额外空格。
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    print("generate_ids---1:", generate_ids)
    print("result---1:", result)
    return result


# 这个函数叫做 generate_constrastive_search，它是用于利用训练好的模型进行对比搜索生成文本的。这个函数接受以下参数：
# model：已经训练好的模型。
# tokenizer：用于将文本转换为模型可理解的输入的工具。
# inputs：模型的输入数据。
# top_k：在每一步生成时，只考虑概率最高的top_k个候选项，然后进行随机抽样。默认为4。
# penalty_alpha：用于惩罚新生成的token与原始输入之间的差异，默认为0.6。
# num_return_sequences：模型返回的序列数，默认为1。
# max_new_tokens：模型生成的最大新token数，即最大生成文本的长度，默认为100。
def generate_constrastive_search(model,
                                 tokenizer,
                                 inputs,
                                 top_k=4,
                                 penalty_alpha=0.6,
                                 num_return_sequences=1,
                                 max_new_tokens=100):
    # 函数首先使用模型的generate方法，根据提供的参数生成文本。注意这里使用了模型的一个特殊的生成方式，这种方式在每一步生成时，只考虑概率最高的top_k个候选项，然后进行随机抽样，同时使用了一个惩罚因子penalty_alpha来惩罚新生成的token与原始输入之间的差异。
    generate_ids = model.generate(inputs.input_ids,
                                  top_k=top_k,
                                  penalty_alpha=penalty_alpha,
                                  num_return_sequences=num_return_sequences,
                                  max_new_tokens=max_new_tokens)
    # 然后，使用tokenizer的batch_decode方法将生成的token ID解码为可读的文本。注意，这里跳过了特殊的token（如填充和开始/结束token），并且不会清理token化产生的额外空格。
    result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
    print("generate_ids---2:", generate_ids)
    print("result---2:", result)

    return result


# gen_output：这是一个列表，其中包含了我们希望打印的内容，每一项都是一段文本。
def print_utils(gen_output):
    # 函数会遍历gen_output列表中的每一项，然后将每一项都打印出来。为了在不同项之间增加一些可视化的分隔，函数在每一项前后都额外打印了一个空行。
    for i in range(len(gen_output)):
        print()
        print(gen_output[i])
        print()


# 然后是prompt_eval这个函数，这个函数prompt_eval的目的是评估和比较基线模型（model_baseline）和微调过的模型（model_fintuned）对于一组提示（prompts）的生成性能。让我们逐行进行解析：
# 输入参数包括：args（命令行参数）、model_baseline（基线模型）、model_fintuned（微调模型）、tokenizer（用于编码和解码的分词器）、device（指定运行模型的设备）、prompts（一组要评估的提示）。
def prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts):
    # 对于prompts中的每一个提示，我们都做以下操作：
    for prompt in prompts:
        # 使用分词器将提示转换为模型所需的输入格式，并将其移至指定的设备上。
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # 打印一条消息表示我们现在正在进行基线模型的生成。
        print("==========Baseline: Greedy=========")
        # 然后，我们调用之前定义的generate函数使用贪婪搜索方法生成文本，并使用print_utils函数打印生成的结果。
        r_base = generate(model_baseline,
                          tokenizer,
                          inputs,
                          num_beams=1,
                          num_return_sequences=args.num_return_sequences,
                          max_new_tokens=args.max_new_tokens)
        print_utils(r_base)
        # 打印一条消息表示我们现在正在进行微调模型的生成。
        print("==========finetune: Greedy=========")
        # 同样地，我们调用generate函数使用贪婪搜索方法生成文本，并使用print_utils函数打印生成的结果。
        r_finetune_g = generate(model_fintuned,
                                tokenizer,
                                inputs,
                                num_beams=1,
                                num_return_sequences=args.num_return_sequences,
                                max_new_tokens=args.max_new_tokens)
        print_utils(r_finetune_g)
        # 注意：在此函数中，贪婪搜索被用作基线方法。然而，该函数还提供了其他几种搜索策略的例子，包括多项式采样、束搜索、束搜索多项式采样、多样性束搜索和对比搜索。这些策略在此函数中都被注释掉了，但你可以根据需要去掉注释，使用这些策略。
        # Note: we use the above simplest greedy search as the baseline. Users can also use other baseline methods,
        # such as beam search, multinomial sampling, and beam-search multinomial sampling.
        # We provide examples as below for users to try.

        # print("==========finetune: Multinomial sampling=========")
        # r_finetune_m = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=1,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_m)
        # print("==========finetune: Beam Search=========")
        # r_finetune_b = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_b)
        # print("==========finetune: Beam-search multinomial sampling=========")
        # r_finetune_s = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         do_sample=True,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_s)
        # print("==========finetune: Diverse Beam Search=========")
        # r_finetune_d = generate(model_fintuned, tokenizer, inputs,
        #                         num_beams=args.num_beams,
        #                         num_beam_groups=args.num_beam_groups,
        #                         num_return_sequences=args.num_return_sequences,
        #                         max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_d)
        # print("==========finetune: Constrastive Search=========")
        # r_finetune_c = generate_constrastive_search(model_fintuned, tokenizer, inputs,
        #                                             top_k=args.top_k,
        #                                             penalty_alpha=args.penalty_alpha,
        #                                             num_return_sequences=args.num_return_sequences,
        #                                             max_new_tokens=args.max_new_tokens)
        # print_utils(r_finetune_c)
        # 最后，打印一条消息表示这个提示的处理已经结束。然后打印两个空行作为分隔。
        print("====================prompt end=============================")
        print()
        print()


# main函数负责解析命令行参数、准备模型和分词器、定义提示，然后使用这些来评估和比较基线模型和微调模型。
def main():
    # 这个main函数是整个脚本的入口点。它首先通过parse_args函数解析命令行参数。然后它设置了运行模型的设备为第一个GPU。
    args = parse_args()

    device = torch.device("cuda:0")
    # 接着，它使用load_hf_tokenizer函数加载分词器，然后使用create_hf_model函数创建基线模型（model_baseline）和微调模型（model_fintuned）
    tokenizer = load_hf_tokenizer(args.model_name_or_path_baseline,
                                  fast_tokenizer=True)

    model_baseline = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_baseline,
                                     tokenizer, None)
    model_fintuned = create_hf_model(AutoModelForCausalLM,
                                     args.model_name_or_path_finetune,
                                     tokenizer, None)
    # 然后，这些模型被移动到指定的设备上。
    print("device :", device)
    print("tokenizer :", tokenizer)
    print("model_baseline :", model_baseline)
    print("model_fintuned :", model_fintuned)

    model_baseline.to(device)
    model_fintuned.to(device)

    # One observation: if the prompt ends with a space " ", there is a high chance that
    # the original model (without finetuning) will stuck and produce no response.
    # Finetuned models have less such issue. Thus following prompts all end with ":"
    # to make it a more meaningful comparison.
    # 在接下来的部分，函数定义了一组用于评估的提示。注意，这里特别指出，如果提示以空格结束，那么没有经过微调的原始模型有可能会陷入停滞并无法产生响应。微调过的模型在这方面表现得更好。因此，这里所有的提示都以冒号":"结束，以使得比较更有意义。
    # 这个脚本支持英文、中文和日文的评估，它通过args.language参数判断用户选择的语言，并根据此选择加载对应的提示。
    if args.language == "English":
        prompts = [
            "Human: Please tell me about Microsoft in a few sentence? Assistant:",
            "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant:",
            "Human: Write a short poem about a wise frog. Assistant:",
            "Human: Who was president of the United States in 1955? Assistant:",
            "Human: How does a telescope work? Assistant:",
            "Human: Why do birds migrate south for the winter? Assistant:"
        ]
    elif args.language == "Chinese":
        prompts = [
            "Human: 请用几句话介绍一下微软? Assistant:",
            "Human: 用几句话向6岁的孩子解释登月。 Assistant:",
            "Human: 写一首关于一只聪明的青蛙的短诗。 Assistant:",
            "Human: 谁是1955年的美国总统? Assistant:", "Human: 望远镜是如何工作的? Assistant:",
            "Human: 鸟类为什么要南迁过冬? Assistant:"
        ]
    elif args.language == "Japanese":
        prompts = [
            "Human: マイクロソフトについて簡単に教えてください。 Assistant:",
            "Human: 6歳児に月面着陸を短い文で説明する。 Assistant:",
            "Human: 賢いカエルについて短い詩を書いてください。 Assistant:",
            "Human: 1955年のアメリカ合衆国大統領は誰? Assistant:",
            "Human: 望遠鏡はどのように機能しますか? Assistant:",
            "Human: 鳥が冬に南に移動するのはなぜですか? Assistant:"
        ]
    # 最后，它调用prompt_eval函数，将所有的参数和提示传递给它，以进行模型评估。
    prompt_eval(args, model_baseline, model_fintuned, tokenizer, device,
                prompts)



if __name__ == "__main__":
    main()
