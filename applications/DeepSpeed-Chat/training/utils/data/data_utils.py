# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Part of the code was adopted from https://github.com/microsoft/Megatron-DeepSpeed/blob/main/megatron/data/dataset_utils.py
"""
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import hashlib  # Python的内置库，提供了一系列散列函数，如MD5、SHA1等。
from itertools import chain  # Python的内置库，提供了一系列用于操作迭代器的函数。
from . import raw_datasets

from pydebug import debuginfo, infoTensor


# 定义好自定义PromptRawDataset后，还需要对其进行“注册”，具体可见下述代码块。
# 这段代码定义了一个名为get_raw_dataset的函数，其主要作用是根据传入的数据集名称dataset_name
# 返回一个适当的PromptRawDataset子类的实例。
# 这个函数有四个参数：dataset_name，output_path，seed和local_rank。
def get_raw_dataset(dataset_name, output_path, seed, local_rank):
    # 在函数中，根据dataset_name的具体值来创建不同的PromptRawDataset子类的实例。
    # 例如，如果dataset_name为"Dahoas/rm-static"，那么就创建一个DahoasRmstaticDataset的实例；
    # 如果dataset_name为"Dahoas/full-hh-rlhf"，那么就创建一个DahoasFullhhrlhfDataset的实例，以此类推。
    if "Dahoas/rm-static" in dataset_name:
        print("yk==dataset_name is:", dataset_name)
        return raw_datasets.DahoasRmstaticDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "Dahoas/full-hh-rlhf" in dataset_name:
        return raw_datasets.DahoasFullhhrlhfDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Dahoas/synthetic-instruct-gptj-pairwise" in dataset_name:
        return raw_datasets.DahoasSyntheticinstructgptjpairwiseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "yitingxie/rlhf-reward-datasets" in dataset_name:
        return raw_datasets.YitingxieRlhfrewarddatasetsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "openai/webgpt_comparisons" in dataset_name:
        return raw_datasets.OpenaiWebgptcomparisonsDataset(
            output_path, seed, local_rank, dataset_name)
    elif "stanfordnlp/SHP" in dataset_name:
        return raw_datasets.StanfordnlpSHPDataset(output_path, seed,
                                                  local_rank, dataset_name)
    elif "pvduy/sharegpt_alpaca_oa_vicuna_format" in dataset_name:
        return raw_datasets.PvduySharegptalpacaoavicunaformatDataset(
            output_path, seed, local_rank, dataset_name)
    elif "wangrui6/Zhihu-KOL" in dataset_name:
        return raw_datasets.Wangrui6ZhihuKOLDataset(output_path, seed,
                                                    local_rank, dataset_name)
    elif "Cohere/miracl-zh-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiraclzhqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "Hello-SimpleAI/HC3-Chinese" in dataset_name:
        return raw_datasets.HelloSimpleAIHC3ChineseDataset(
            output_path, seed, local_rank, dataset_name)
    elif "mkqa-Chinese" in dataset_name:
        return raw_datasets.MkqaChineseDataset(output_path, seed, local_rank,
                                               "mkqa")
    elif "mkqa-Japanese" in dataset_name:
        return raw_datasets.MkqaJapaneseDataset(output_path, seed, local_rank,
                                                "mkqa")
    elif "Cohere/miracl-ja-queries-22-12" in dataset_name:
        return raw_datasets.CohereMiracljaqueries2212Dataset(
            output_path, seed, local_rank, dataset_name)
    elif "lmqg/qg_jaquad" in dataset_name:
        return raw_datasets.LmqgQgjaquadDataset(output_path, seed, local_rank,
                                                dataset_name)
    elif "lmqg/qag_jaquad" in dataset_name:
        return raw_datasets.LmqgQagjaquadDataset(output_path, seed, local_rank,
                                                 dataset_name)
    # 如果dataset_name是"local/jsonfile"，则会检查在路径chat_path + '/data/train.json'
    # 和chat_path + '/data/eval.json'下是否存在文件。如果存在，则创建一个LocalJsonFileDataset的实例；
    # 如果不存在，则抛出一个RuntimeError异常。
    elif "local/jsonfile" in dataset_name:
        chat_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.path.pardir,
                         os.path.pardir, os.path.pardir))
        if not (os.path.isfile(chat_path + '/data/train.json')
                and os.path.isfile(chat_path + '/data/eval.json')):
            raise RuntimeError(
                f"Please check both the train.json and eval.json files in your applications/DeepSpeed-Chat/data directory."
            )
        return raw_datasets.LocalJsonFileDataset(output_path, seed, local_rank,
                                                 dataset_name, chat_path)


    # 将自定义的PromptRawDataset在此处进行注册
    # 届时在传参“--data_path”中赋值“custom”即可读取到相应的数据集
    elif "custom" in dataset_name:
        return raw_datasets.CustomDataset(output_path, seed,
                                          local_rank, dataset_name)

    # 至此完成自定义数据集的设置。理论上来说，只要实例函数能完全按照注释要求对原始数据进行处理，
    # 那么后续的数据流基本也无需再进行任何额外修改也能顺畅运行了。
    else:
        # 如果dataset_name没有在以上的所有条件中匹配到，那么函数也会抛出一个RuntimeError异常，表示没有为这个数据集的配置。
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


# 这个函数的作用是生成一个大小为size的乱序索引数组，它接受两个参数：seed和size。
def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)  # 创建一个NumPy的随机状态生成器对象np_rng，seed是随机种子，确定了随机数的生成序列。
    dtype_ = np.uint32  # 设置其为NumPy的uint32类型，这是一个无符号32位整数类型。
    if size >= (np.iinfo(np.uint32).max - 1):  # 如果size大于或等于uint32的最大值减一，这里减一是为了防止可能的溢出。
        dtype_ = np.int64  # 则将dtype_改为int64，这是一个64位的有符号整数类型。
    shuffle_idx = np.arange(start=0, stop=size, step=1,
                            dtype=dtype_)  # 创建一个由0开始，步长为1，到size结束（不包含size），并且数据类型为dtype_的等差数列，将其赋值给shuffle_idx。
    np_rng.shuffle(shuffle_idx)  # 使用np_rng随机状态生成器对shuffle_idx进行随机排列，这样就打乱了shuffle_idx的顺序。

    print("len of shuffle_idx is:", len(shuffle_idx))
    # print("shuffle_idx is:", shuffle_idx)
    # shuffle_idx is: [6503 4944 5285 ... 1318  723 2863]
    return shuffle_idx  # 返回乱序后的shuffle_idx。


# 这个函数主要是根据提供的参数分割数据集，并生成一个分割索引。
# 它首先检查索引文件是否存在，如果不存在，则生成分割索引，并保存到文件。
# 然后，它从文件中加载索引，并返回索引列表。
# 它接受8个参数：local_rank、输出路径、数据集名称、种子、split_name、data_split、split_index和数据大小。
def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    # 根据给定参数生成索引文件名。
    '''
   608 10月  9 15:41 Dahoas_rm_static_seed1234_eval_2,4,4_1.npy
   1728 10月  9 15:41 Dahoas_rm_static_seed1234_train_2,4,4_2.npy
    '''
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"

    # reindex each time when using local jsonfile since it's more likely to get modified
    # 如果索引文件不存在，或者数据集名为'jsonfile'，则执行下面的操作。
    if (not os.path.isfile(index_file_name)) or (dataset_name == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]  # 将data_split（以逗号分隔的字符串）分割为浮点数列表。
        splits_sum = sum(splits)  # 计算所有分割的总和。
        splits = [split / splits_sum for split in splits]  # 将每个分割值归一化，使其和为1。
        splits_index = [0]  # 初始化分割索引列表，起始值为0
        for index, split in enumerate(splits):  # 遍历分割列表。
            # 将新的索引（当前索引加上归一化的分割值与数据大小的乘积）添加到索引列表。
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        # 计算最后一个索引与数据大小的差值。
        diff = splits_index[-1] - data_size
        # 遍历除第一个外的所有索引。
        for index in range(1, len(splits_index)):
            # 将差值从每个索引中减去，以确保最后一个索引等于数据大小。
            splits_index[index] -= diff
        # 断言最后一个索引等于数据大小
        assert splits_index[-1] == data_size

        # 生成一个乱序的索引。
        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):  # 遍历每个分割。
            # 根据给定参数生成乱序索引分割文件名。
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            # 提取乱序索引的一个分割。
            shuffle_idx_split = shuffle_idx[
                                splits_index[split_i]:splits_index[split_i + 1]]
            # 将乱序索引分割保存到文件。
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    # 加载索引文件。
    index = np.load(index_file_name, allow_pickle=True)
    #print("index is:", index)
    print("len of index is:", len(index))

    # 将索引数组转换为列表并返回。
    return index.tolist()


# 这是一个自定义的PromptDataset类，它继承自torch.utils.data.Dataset。
# 这是一个数据集类，通常被用于PyTorch中数据的加载和预处理。
class PromptDataset(Dataset):
    # 类的构造函数，它接受五个参数：
    # prompt_dataset、chosen_dataset、reject_dataset、pad_token_id和train_phase。
    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        debuginfo(prj='ds-chat', info=self.__class__.__name__)

        super().__init__()  # 调用父类torch.utils.data.Dataset的构造函数。

        self.prompt_dataset = prompt_dataset  # 将传入的参数赋值给类的成员变量。
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        # 定义类的__len__方法，它返回数据集的长度。
        # 这是PyTorch数据集的必要方法。
        length = len(self.chosen_dataset)  # 初始设定数据集长度为chosen_dataset的长度。
        if self.train_phase == 3:
            length = len(self.prompt_dataset)  # 如果训练阶段为3，则数据集长度设定为prompt_dataset的长度。
        return length  # 返回计算得出的数据集长度。

    # 定义类的__getitem__方法，它接受一个参数idx，返回索引idx处的数据。
    # 这是PyTorch数据集的必要方法。
    def __getitem__(self, idx):
        # 如果训练阶段为1，则返回一个字典，包含input_ids、attention_mask和labels，它们都来自chosen_dataset的索引idx处。
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        # 如果训练阶段为2，则返回来自chosen_dataset和reject_dataset的input_ids和attention_mask。
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        # 如果训练阶段为3，则返回来自prompt_dataset的input_ids、attention_mask和pad_token_id
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"], \
                self.pad_token_id


'''
0.2.3.2 阶段数据集处理过程
UML时序图(10-12)
这部分处理得到的数据形式，基本接近于数据传入阶段模型前的最终形式，
因此通过理解这部分的数据处理过程，可以直接了解到模型所需要的输入形式。

此处的处理部分很大程度依赖于原先所定义的PromptRawDataset实例函数，由此可见，只要正确编写实例函数，
后续过程基本也不会出现什么问题。
流程大致就是取出对应阶段所需的格式数据，然后使用tokenizer进行处理，综上所述：

phase1模型所需的输入数据为chosen_sentence的input_ids及attention_mask；
phase2模型所需的输入数据为chosen_sentence和reject_sentence的input_ids及attention_mask；
phase3模型所需的输入数据为promt的input_ids及attention_mask。
'''
# 这是一个名为create_dataset_split的函数，它的功能是根据给定的训练阶段（train_phase），创建并返回相应的数据集分割。
# 具体来说，它为每个训练阶段生成不同的数据集列表，并将它们放入PromptDataset对象中。
# 函数接受6个参数：当前数据集(current_dataset)、原始数据集(raw_dataset)、训练阶段(train_phase)、
# 分词器(tokenizer)、会话结束标记(end_of_conversation_token)和最大序列长度(max_seq_len)。
def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    # 将根据不同的阶段（train_phase）对数据集进行处理，主要是调用原先在PromptRawDataset类中定义的实例函数来实现。
    # 创建三个空的列表，用于存储对话提示（prompt_dataset）、选定的对话（chosen_dataset）和被拒绝的对话（reject_dataset）。
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    # 如果训练阶段为1，则将接受的对话进行分词并添加到chosen_dataset中。
    if train_phase == 1:  #需要刪除data_files才可以
        debuginfo(prj='ds-chat', info="train_phase == 1")
        # 2.1数据处理：
        # ● 只需要获得训练集和验证集即可，也可以进行采样；
        # ● 接着，读取的数据中，获取prompt和chosen两个字段：

        # 因为phase1只需要用到chosen数据，所以只取chosen进行处理
        # 遍历当前数据集。
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            # 从原始数据集中获取对话提示和接受的对话。
            # 获取chosen_sentence，即是将prompt和chosen拼接起来形成完整对话
            # 具体样例可参照“数据格式基本概念”中的样例
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response

            #print("chosen_sentence--ph1:", chosen_sentence)

            # 如果接受的对话不为空，则将其分词并添加到chosen_dataset中。
            if chosen_sentence is not None:
                # 在对话末尾加入对话终止符
                # end_of_conversation_token表示每个对话的终止符，可以用“<|endoftext|>”表示
                chosen_sentence += end_of_conversation_token
                # 使用tokenizer处理chosen_sentence，采取截断truncation
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                # 去掉batch维度
                chosen_token["input_ids"] = chosen_token["input_ids"].squeeze(
                    0)
                chosen_token["attention_mask"] = chosen_token[
                    "attention_mask"].squeeze(0)
                # 存储tokenize结果至列表chosen_dataset
                chosen_dataset.append(chosen_token)
                #print("chosen_token--ph1:", chosen_token)
            '''
            chosen_sentence--ph1:
            
            Human: Can you tell me how often I should be changing my sheets?
            
            Assistant: A good rule of thumb is to change the sheets on your bed once a week.  It can depend on how many people sleep in your bed, and how many wet spots and smells accumulate on your sheets.
            chosen_token--ph1: {
            'input_ids': tensor([    2, 50118, 50118, 33837,    35,  2615,    47,  1137,   162,   141,
                        ...
                        2,     2,     2,     2,     2,     2,     2,     2]), 
            'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        ...
                    0, 0, 0, 0, 0, 0, 0, 0])}
            '''

            #print("T chosen_token['input_ids']-1:", infoTensor(chosen_token['input_ids']))
            #print("T chosen_token['attention_mask']-1:", infoTensor(chosen_token['attention_mask']))
            # T chosen_token['input_ids']-1: _Size([128])_int64_cpu_        #only ph1
            # T chosen_token['attention_mask']-1: _Size([128])_int64_cpu_   #only ph1

        #print("chosen_dataset--ph1:", chosen_dataset) ===就是 chosen_token 的 list
        print("len of chosen_dataset--ph1:", len(chosen_dataset))

        # ● 此时，一条样本可以表示为prompt+chosen，
        # 中间会插入一些用于对话的标记，例如“Human: ”、“Assistant: ”、“<|endoftext|>”等。

    # 如果训练阶段为2，则将接受和被拒绝的对话都进行分词并分别添加到chosen_dataset和reject_dataset中。
    elif train_phase == 2:
        debuginfo(prj='ds-chat', info="train_phase == 2")
        # phase2需要用到chosen_sentence和reject_sentence
        # 所以需要对两者都进行处理

        # 3.1数据处理：
        # ● 读取训练集和验证集用来训练偏好模型；
        # ● 此时需要读取prompt、chosen和rejected三个字段数据，每一条数据是一个pairwise
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            # 获取chosen_sentence，即是将prompt和chosen拼接起来形成完整对话
            # 具体样例可参照“数据格式基本概念”中的样例
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response

            # 获取reject_sentence，即是将prompt和rejeced拼接起来形成完整对话
            # 具体样例可参照“数据格式基本概念”中的样例
            reject_sentence = raw_dataset.get_prompt_and_rejected(
                tmp_data)  # the accept response

            # print("chosen_sentence--ph2:", chosen_sentence)
            # print("reject_sentence--ph2:", reject_sentence)

            if chosen_sentence is not None and reject_sentence is not None:
                # 在对话末尾加入对话终止符
                chosen_sentence += end_of_conversation_token  # the accept response
                reject_sentence += end_of_conversation_token

                # 使用tokenizer处理，采取截断truncation
                chosen_token = tokenizer(chosen_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                reject_token = tokenizer(reject_sentence,
                                         max_length=max_seq_len,
                                         padding="max_length",
                                         truncation=True,
                                         return_tensors="pt")
                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]

                # 存储tokenize结果至列表chosen_dataset
                chosen_dataset.append(chosen_token)

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]

                # 存储tokenize结果至列表reject_dataset
                reject_dataset.append(reject_token)

                # print("reject_token--ph2:", reject_token)
                # print("chosen_token--ph2:", chosen_token)

            '''
            chosen_sentence--ph2:
    
                Human: are crunches or sit-ups better?
                
                Assistant: I would recommend both! They can help you stay healthy and are also helpful if you want to lose weight.
                
            reject_sentence--ph2:
            
                Human: are crunches or sit-ups better?
                
                Assistant: What are you looking for exactly?
            
            reject_token--ph2: {
                'input_ids': tensor([[    2, 50118, 50118, 33837,    35,    32,  3977,   879,  5559,    50,
                        ...
                         2,     2,     2,     2,     2,     2,     2,     2]]), 
                'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        ...
                     0, 0, 0, 0, 0, 0, 0, 0]])}
                     
            chosen_token--ph2: {
                'input_ids': tensor([[    2, 50118, 50118, 33837,    35,    32,  3977,   879,  5559,    50,
                        ...
                         2,     2,     2,     2,     2,     2,     2,     2]]), 
                'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    ...
                     0, 0, 0, 0, 0, 0, 0, 0]])}
            '''
            # print("T reject_token['input_ids']--ph2:", infoTensor(reject_token['input_ids']))
            # print("T reject_token['attention_mask']--ph2:", infoTensor(reject_token['attention_mask']))
            # print("T chosen_token['input_ids']--ph2:", infoTensor(chosen_token['input_ids']))
            # print("T chosen_token['attention_mask']--ph2:", infoTensor(chosen_token['attention_mask']))
            ''' only ph2
            T reject_token['input_ids']--ph2: _Size([1, 128])_int64_cpu_
            T reject_token['attention_mask']--ph2: _Size([1, 128])_int64_cpu_
            T chosen_token['input_ids']--ph2: _Size([1, 128])_int64_cpu_
            T chosen_token['attention_mask']--ph2: _Size([1, 128])_int64_cpu_
            '''

        #print("reject_dataset--ph2:", reject_dataset)   就是 reject_token 的 list
        print("len of reject_dataset--ph2:", len(reject_dataset))

        # print("chosen_dataset--ph2:", chosen_dataset)  就是 chosen_token 的 list
        print("len of chosen_dataset--ph2:", len(chosen_dataset))


    # 如果训练阶段为3，则将对话提示进行分词并添加到prompt_dataset中。
    elif train_phase == 3:
        debuginfo(prj='ds-chat', info="train_phase == 3")
        # phase3用到prompt，prompt将被用来生成经验数据

        # 4.1数据处理
        # 在第三阶段，可以选择监督训练数据和无监督数据。
        # ● 监督数据：此时只有prompt，没有chosen和rejected input。
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            # 直接获取prompt
            # 具体样例可参照“数据格式基本概念”中的样例
            prompt = raw_dataset.get_prompt(tmp_data)
            #print("prompt--ph3:", prompt)

            if prompt is not None:
                prompt_token = tokenizer(prompt, return_tensors="pt")
                prompt_token["input_ids"] = prompt_token["input_ids"]
                prompt_token["attention_mask"] = prompt_token["attention_mask"]
                for key_word in ["input_ids", "attention_mask"]:
                    # 获取当前文本token的实际长度
                    length = prompt_token[key_word].size()[-1]
                    # phase3此处的max_seq_len其实是max_prompt_len，默认只有256
                    if length > max_seq_len:
                        # 如果当前文本token长度比max_prompt_len还长
                        # 那么就截断文本前面的部分，保留后面max_prompt_len长度的部分文本
                        # 然后将token进行flip（翻转/倒序），之后在data_collator中再将其flip回来

                        # 先将正常的token序列的顺序倒序排列，（会在datacollator中再次倒序恢复原始排列）
                        y = prompt_token[key_word].squeeze(0)[length -
                                                              (max_seq_len -
                                                               1):].flip(0)
                    else:
                        # 将token进行flip（翻转/倒序），之后在data_collator中再将其flip回来

                        # 先将正常的token序列的顺序倒序排列，（会在datacollator中再次倒序恢复原始排列）
                        y = prompt_token[key_word].squeeze(0).flip(0)
                    prompt_token[key_word] = y
                prompt_dataset.append(prompt_token)

                #print("prompt_token--ph3:", prompt_token)
                '''
                prompt--ph3:
                
                Human: Can you recommend some ways to propose marriage to my girlfriend?
                
                Assistant: Sure, how about if you had a romantic vacation planned, and then when you got there you surprised her with a ring?
                
                Human: That's a great suggestion! Should I propose in a specific location?
                
                Assistant: Definitely in a location that means something to the two of you, and you've found a place that represents something important to you both, like a spot where you both first kissed or talked about marriage.
                
                Human: Good plan. We could go to Greece, where we first met.
                
                Assistant: Oh, that sounds perfect!
                
                Human: Is it very expensive to buy an engagement ring?
                
                Assistant: It might be if you aren't very careful about the type of ring and where you buy it.  For instance, if the ring isn't sized right and the wrong type, it will need to be re-sized later.  And the larger the diamond, the more you'll pay.  So it's best to choose a simple ring, if it's what the two of you really want.  Keep in mind that you don't need a diamond ring to be engaged!
                
                Human: Right, got it, thanks.
                
                Assistant:
                prompt_token--ph3: {
                    'input_ids': tensor([   35, 46184, 50118, 50118,     4,  2446,     6,    24,   300,     6,
                        ...
                         6096,   127,     7,  3397, 15393]), 
                    'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        ...
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
                '''
                # print("T prompt_token['input_ids']--H:", infoTensor(prompt_token['input_ids']))
                # print("T prompt_token['attention_mask']--H:", infoTensor(prompt_token['attention_mask']))
                '''
                大小可变, only ph3 z1,z1
                T prompt_token['input_ids']--H: _Size([134])_int64_cpu_
                T prompt_token['attention_mask']--H: _Size([134])_int64_cpu_
                '''

        #print("prompt_dataset--ph3:", prompt_dataset) 就是prompt_token的list
        print("len of prompt_dataset--ph3:", len(prompt_dataset))

    # 返回PromptDataset实例，该实例相当于torch中的Dataset，可供DataLoader调用
    # 创建一个新的PromptDataset对象，并返回。这个对象包含了对话提示、接受的对话和被拒绝的对话的数据集，
    # 以及分词器的填充标记ID和训练阶段。
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


# 这段代码定义了一个函数 create_dataset，主要负责创建训练数据集和评估数据集，具体的功能细节如下：
def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len):
    debuginfo(prj='ds-chat', info=f"train_phase {train_phase}")

    # 调用 get_raw_dataset 函数，该函数根据提供的数据集名称、输出路径、随机种子和local_rank等参数，
    # 从各种预定义的数据集中获取所需的原始数据集。
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()  # 从原始数据集中获取训练数据。
    #print("raw_dataset is:", raw_dataset)
    #print("train_dataset---A is:", train_dataset)
    '''
        
    raw_dataset is: <utils.data.raw_datasets.DahoasRmstaticDataset object at 0x7fe83804ed00>
    train_dataset---A is: Dataset({
        features: ['prompt', 'response', 'chosen', 'rejected'],
        num_rows: 7000
    })  
        '''

    #  获取训练数据集的索引，涉及数据的分割。
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset))

    # 根据上一步获取的索引，创建训练数据的子集。
    train_dataset = Subset(train_dataset, train_index)
    print("len of train_index is:", len(train_index))
    # print("train_index is:", train_index)
    # print("train_dataset---B is:", train_dataset)
    '''
    train_index is: [869, 1971, 6162, 4194, 1508, 2043, 3775,...]
    train_dataset---B is: <torch.utils.data.dataset.Subset object at 0x7fe7f80dac40>``
    '''

    # 调用 create_dataset_split 函数对上一步获得的数据子集进行进一步处理，
    # 这可能包括对文本的标记化(tokenization)，并且创建一个PromptDataset 对象。
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    # print("train_dataset---C is:", train_dataset)
    # train_dataset---C is: <utils.data.data_utils.PromptDataset object at 0x7fb5680452b0>

    # 是用于创建评估数据集的，步骤与训练数据集的创建基本相同。
    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset))

    # print("eval_index is:", eval_index)
    # eval_index is: [1551, 1476, 40, 2157, 1317, 1711, 712, 2070,
    print("len of eval_index is:", len(eval_index))

    # print("eval_dataset---A is:", eval_dataset)


    eval_dataset = Subset(eval_dataset, eval_index)
    # print("eval_dataset---B is:", eval_dataset)

    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    # print("eval_dataset---C is:", eval_dataset)

    '''

    eval_dataset---A is: Dataset({
        features: ['prompt', 'response', 'chosen', 'rejected'],
        num_rows: 3000
    })
    eval_dataset---B is: <torch.utils.data.dataset.Subset object at 0x7fb5680505b0>
    eval_dataset---C is: <utils.data.data_utils.PromptDataset object at 0x7fb5c37cd2e0>
    '''

    return train_dataset, eval_dataset


# 这里需要额外注意一点，create_dataset_split 这个函数传入的参数是 train_phase，
# 而不是像 get_raw_dataset_split_index 函数那样传入 train_phase-1，这是为什么？
# 这是因为train_phase用来标识在模型训练过程中的哪个阶段，是一个直接的指示符，与数据分割或索引无关。
# 它用来在create_dataset_split函数中确定当前处于哪个训练阶段，并根据不同的训练阶段对数据进行不同的处理。
# 而在调用get_raw_dataset_split_index函数时，传入的是train_phase - 1，这是因为在这个函数中，
# 我们需要根据当前训练阶段的前一个阶段（由train_phase - 1表示）的数据分割或索引情况，来决定如何对当前阶段的数据进行分割或索引。
# create_dataset_split主要关注如何根据训练阶段对数据进行处理，
# 而get_raw_dataset_split_index主要关注如何根据前一个训练阶段的数据分割或索引情况，对当前阶段的数据进行分割或索引。

# 回收本节的开头，我们解析create_prompt_dataset函数：
# 这个函数的主要目的是创建一个包含训练和评估数据集的“提示”数据集，并将这两个数据集保存在指定的文件中。具体来说：
def create_prompt_dataset(local_rank,
                          data_path,
                          data_split,
                          output_path,
                          train_phase,
                          seed,
                          tokenizer,
                          max_seq_len,
                          end_of_conversation_token="<|endoftext|>",
                          sft_only_data_path=[],
                          reload=False):
    debuginfo(prj='ds-chat', info=f"train_phase {train_phase}")

    """
    Creates the prompt dataset
    """
    # os.makedirs(output_path, exist_ok=True): 创建输出目录，如果目录已经存在则不会引发异常。
    os.makedirs(output_path, exist_ok=True)
    # 构造文件名，这个文件名包含了很多有关数据集和模型的信息，如数据路径、数据分割、训练阶段、
    # 随机种子、tokenizer的名称、最大序列长度等。然后将这个文件名哈希化，以避免文件名过长。
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    # fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = f"{fname}_ph{train_phase}_tokenizer{tokenizer_name}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    assert(len(fname)) < 100, len(fname)

    # 调试中取消哈希
    # fname = hashlib.sha256(fname.encode()).hexdigest()  # hash the file name to avoid too long file name

    # 构造训练数据集和评估数据集的文件路径。
    train_fname = f"../../traindata_{fname}.pt"
    eval_fname = f"../../evaldata_{fname}.pt"

    print("train_fname is", train_fname)
    print("train_fname is", train_fname)
    '''
    要想看到dataset创建过程，就要删除， ph1,2,3都有！
    原来位置和文件名样式
train_fname is /tmp/data_files//traindata_aa981ebbdc26ba0c4e46b123a94edae66e2c058a407c8ff11ea6c5bbe67c27cd.pt
train_fname is /tmp/data_files//traindata_aa981ebbdc26ba0c4e46b123a94edae66e2c058a407c8ff11ea6c5bbe67c27cd.pt

train_fname is /tmp/data_files/traindata_e7b11df4f76290627ffa57589ebd268d59ce98a2126ca343b4b6b22e9bc44c80.pt
train_fname is /tmp/data_files/traindata_e7b11df4f76290627ffa57589ebd268d59ce98a2126ca343b4b6b22e9bc44c80.pt
    '''

    # 检查训练数据集和评估数据集的文件是否都已经存在，如果存在，则表示缓存已经找到，否则表示需要创建缓存。
    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    # 如果当前进程是主进程（local_rank <= 0）并且需要创建缓存，就执行以下操作。
    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        # 如果只有一个数据集，直接调用create_dataset函数创建训练数据集和评估数据集。
        debuginfo(prj='ds-chat', info="只有一个数据集")
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank, data_path[0], data_split, output_path, train_phase,
                seed, tokenizer, end_of_conversation_token, max_seq_len)
        else:  # Blending datasets.
            debuginfo(prj='ds-chat', info="多个数据集")
            # 如果有多个数据集，对每个数据集都调用create_dataset函数，并把得到的训练数据集和评估数据集添加到对应的列表中，

            train_datasets = []
            eval_datasets = []
            train_size = 0
            eval_size = 0
            for d_path in data_path:
                train_dataset, eval_dataset = create_dataset(
                    local_rank, d_path, data_split, output_path, train_phase,
                    seed, tokenizer, end_of_conversation_token, max_seq_len)
                train_datasets.append(train_dataset)
                eval_datasets.append(eval_dataset)
                train_size += len(train_dataset)
                eval_size += len(eval_dataset)
            # 然后使用ConcatDataset和Subset函数合并数据集。
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        # 如果当前是第一阶段的训练（SFT）并且指定了仅用于SFT的数据集，那么对这些数据集执行类似的操作，
        # 然后把得到的训练数据集和评估数据集添加到原有的数据集中。
        if train_phase == 1 and sft_only_data_path:
            debuginfo(prj='ds-chat', info="train_phase == 1 and sft_only_data_path")
            sft_train_datasets = []
            sft_eval_datasets = []
            sft_train_size = 0
            sft_eval_size = 0
            for sft_path in sft_only_data_path:
                sft_train_dataset, sft_eval_dataset = create_dataset(
                    local_rank,
                    sft_path,
                    "10,0,0",
                    output_path,
                    train_phase,
                    seed,
                    tokenizer,
                    end_of_conversation_token,
                    max_seq_len,
                )
                sft_train_datasets.append(sft_train_dataset)
                sft_eval_datasets.append(sft_eval_dataset)
                sft_train_size += len(sft_train_dataset)
                sft_eval_size += len(sft_eval_dataset)
            if sft_train_datasets:  # Check if sft_train_datasets is not empty
                sft_train_dataset = ConcatDataset(sft_train_datasets)
                train_dataset = ConcatDataset(
                    [train_dataset, sft_train_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(train_dataset))
                train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            if sft_eval_datasets:  # Check if sft_eval_datasets is not empty
                sft_eval_dataset = ConcatDataset(sft_eval_datasets)
                eval_dataset = ConcatDataset([eval_dataset, sft_eval_dataset])
                shuffle_idx = get_shuffle_idx(seed, len(eval_dataset))
                eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # 把训练数据集和评估数据集保存到对应的文件中。
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)

    # 在多进程环境中，确保所有进程都完成了数据集的保存操作。
    torch.distributed.barrier()

    return torch.load(train_fname), torch.load(eval_fname)


'''
输入的data为一个batch的数据列表，其中的 每个元素 为一对chosen-rejected数据：
	(
	 chosen_sentence_input_ids, 
	 chosen_sentence_attention_mask,
	 reject_sentence_input_ids,
	 reject_sentence_attention_mask
	)

每组数据的第0个元素和第2个元素为input_ids，第1个元素和第3个元素为attention_mask。

输出的batch为字典：{“input_ids”: tensor([...]), "attention_mask": tensor([...])}
并且字典值中chosen位于前半部分，rejected位于后半部分：
	{
	"input_ids": [
				  chosen_sentence_1_input_ids,
				  chosen_sentence_2_input_ids,
				  ...,
				  reject_sentence_1_input_ids,
				  reject_sentence_2_input_ids,
				  ...
				 ]
	"attention_mask": [
					   chosen_sentence_1_attention_mask,
					   chosen_sentence_2_attention_mask,
					   ...,
					   reject_sentence_1_attention_mask,
					   reject_sentence_2_attention_mask,
					   ...
					  ]

	}
后续输入模型后，直接将数据切分出前半部分和后半部分进行并列，即可获得对应的chosen-rejected数据对。
'''

# 3.2 DataCollator
# 给定一个batch，其包含batch_size个chosen examples和rejected examples，将其进行拆分，具体操作如下：
class DataCollatorReward:

    def __call__(self, data):
        """
                对dataloader取到的数据 data 进一步整理，将数据整理成batch输入形式
                入参 data 的具体样式可见下个代码块
        """
        batch = {}

        # f[0]是chosen input ids，f[2]是rrejected input ids
        # 该操作是指，先给定N个examples，转换为2*N个样本。
        # 前N个为chosen input ids，后N个为rejected input ids

        """
        f为data中的1个tuple，tuple的第0个元素和第2个元素
        分别为chosen_sentence和reject_sentence的input_ids
        """
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)

        """
        f为data中的1个tuple，tuple的第1个元素和第3个元素
        分别为chosen_sentence和reject_sentence的attention_mask
        """
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)

        """batch的具体样式可见下个代码块"""
        #print("batch--D:", batch)
        '''
        batch--D: {
            'input_ids': tensor([[    2, 50118, 50118,  ...,     2,     2,     2],
        ...
        [    2, 50118, 50118,  ...,    24,    35, 50118]]), 
            'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        ...
        [1, 1, 1,  ..., 1, 1, 1]])}
        '''

        # print("T batch['input_ids']--F:", infoTensor(batch['input_ids']))
        # print("T batch['attention_mask']--F:", infoTensor(batch['attention_mask']))
        '''
        T batch['input_ids']--F: _Size([16, 128])_int64_cpu_
        T batch['attention_mask']--F: _Size([16, 128])_int64_cpu_
        '''

        return batch


# 4.2 DataCollator
# 针对监督数据，需要进行处理：
class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        debuginfo(prj='ds-chat', info=self.__class__.__name__)
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
        debuginfo(prj='ds-chat', info=self.__class__.__name__)
        batch = {}
        pad_token_id = data[-1][-1]

        prompt = pad_sequence([f[0] for f in data],
                              padding_value=pad_token_id,
                              batch_first=True)
        prompt_mask = pad_sequence([f[1] for f in data],
                                   padding_value=0,
                                   batch_first=True)

        ### make sure the final ouput is a seqence of 2**?
        length = prompt.size()[-1]
        pad_length = self.max_token_len - length
        if pad_length > 0:
            batch["prompt"] = F.pad(prompt,
                                    pad=(0, pad_length),
                                    mode='constant',
                                    value=pad_token_id)
            batch["prompt_att_mask"] = F.pad(prompt_mask,
                                             pad=(0, pad_length),
                                             mode='constant',
                                             value=0)
        else:
            batch["prompt"] = prompt
            batch["prompt_att_mask"] = prompt_mask
        batch["prompt"] = batch["prompt"].flip(1)
        batch["prompt_att_mask"] = batch["prompt_att_mask"].flip(1)

        # print("batch-RLHF:", batch)
        '''
        batch-RLHF: {
            'prompt': tensor([[    2,  1667,    14,  ..., 50118, 46184,    35],
            ...
        [    2,    80,  3678,  ..., 50118, 46184,    35]]), 
            'prompt_att_mask': tensor([[0, 1, 1,  ..., 1, 1, 1],
            ...
        [0, 1, 1,  ..., 1, 1, 1]])}
        '''

        # print("T batch['prompt']--RLHF:", infoTensor(batch['prompt']))
        # print("T batch['prompt_att_mask']--RLHF:", infoTensor(batch['prompt_att_mask']))
        ''' only ph3
        T batch['prompt']--RLHF: _Size([4, 256])_int64_cpu_
        T batch['prompt_att_mask']--RLHF: _Size([4, 256])_int64_cpu_
        '''

        return batch


# ● 无监督数据：只有文本，并进行group：
def get_unsupervised_data(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
                [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    return train_dataset


'''
3.3.4 PPO训练数据管理-MiniDataset
最开始的时候载入过一次Dataset（见3.3.1），但刚开始载入的Dataset针对的是全部训练数据的管理，
而此时使用的MiniDataset主要针对PPO训练迭代所使用的数据进行管理。PPO训练前的数据管理流程可以理解为：

    1 Dataloader从Dataset中取出1个prompt_batch的无监督数据和1个prompt_batch的prompt数据；
    2 使用1个prompt_batch的prompt数据进行经验采集，将得到1个prompt_batch的经验数据；
    3 1个prompt_batch的无监督数据、1个prompt_batch的经验数据将被送入各自的MiniDataset实例进行管理：
      1个prompt_batch将被分成数个ppo_batch，供PPO训练进行数次迭代。

上述第3步就是MiniDataset所要做的事，其函数方法分别执行了：

    1 add()：获取batch（prompt_batch）数据；

    2 seperate()：细分为ppo_batch数据；

    3 free():清空获取到的batch数据并返回ppo_batch数据。

'''

class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        '''
        :param max_size: batch数。通常此处指“用于给actor做生成的prompt的batch数（注意是batch数不是batch_size）”。

        :param small_batch_size: batch size。通常此处指“PPO训练的batch_size”。
        '''
        debuginfo(prj='ds-chat', info=self.__class__.__name__)

        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        debuginfo(prj='ds-chat', info=self.__class__.__name__)

        # 维护1个small_dataset
        small_dataset = []

        # 从self.dataset中逐个取batch
        for large_batch in self.dataset:
            # 判断batch的数据类型（列表 / 元组 / 字典），
            # 根据数据类型取其batch_size，赋值给large_size
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)

            '''
            以下部分代码略微抽象，需要举例说明
            - 比如prompt的batch_size设置为3，PPO训练用的batch_size设置为4，
            则最后能取来用、存入small_dataset的也就只有3条数据，因为生成用的dataloader只采样出了3条，最多也就只有3条。

            - 比如prompt的batch_size设置为5，PPO训练用的batch_size设置为4，
            则最后能取来用、存入small_dataset的就是2组数据（第1组为idx0,idx1,idx2,idx3共4条数据、第2组为idx4共1条数据）。

            - 比如prompt的batch_size设置为9，PPO训练用的batch_size设置为4，
            则最后能取来用、存入small_dataset的就是3组数据（[0,1,2,3],[4,5,6,7],[8]）。
            '''
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append(
                        [x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({
                        k: v[i:i + self.small_batch_size]
                        for k, v in large_batch.items()
                    })
                else:
                    small_dataset.append(large_batch[i:i +
                                                       self.small_batch_size])
        # 清空self.dataset
        self.free()

        # 返回最终取用的数据，该ppo_batch数据将用于ppo训练迭代
        return small_dataset

    def add(self, data):
        """
        在最开始的时候可以传参预设“生成X个batch再进行PPO训练”，
        此处的max_size就是其中的X，
        如果少于max_size则将batch数据加入至MiniDataset中，
        直至达到max_size个batch
        """
        debuginfo(prj='ds-chat', info=self.__class__.__name__)

        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                """
                seperate()主要实现了
                1. 在batch的基础上，再细分ppo_batch并返回
                2. 清空MiniDataset中的数据
                """
                return self.seperate()
            else:
                return None
        else:
            raise ValueError(
                "The dataset is full but we did not stop it. There is a bug in the code."
            )

    def free(self):
        # 清空self.dataset中的数据
        self.dataset = []
