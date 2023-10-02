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
import hashlib
from itertools import chain
from . import raw_datasets

#定义好自定义PromptRawDataset后，还需要对其进行“注册”，具体可见下述代码块。
def get_raw_dataset(dataset_name, output_path, seed, local_rank):

    if "Dahoas/rm-static" in dataset_name:
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
        raise RuntimeError(
            f"We do not have configs for dataset {dataset_name}, but you can add it by yourself in raw_datasets.py."
        )


def get_shuffle_idx(seed, size):
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def get_raw_dataset_split_index(local_rank, output_path, dataset_name, seed,
                                split_name, data_split, split_index,
                                data_size):
    index_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_index}.npy"
    # reindex each time when using local jsonfile since it's more likely to get modified
    if (not os.path.isfile(index_file_name)) or (dataset_name == 'jsonfile'):
        splits = [float(s) for s in data_split.split(',')]
        splits_sum = sum(splits)
        splits = [split / splits_sum for split in splits]
        splits_index = [0]
        for index, split in enumerate(splits):
            splits_index.append(splits_index[index] +
                                int(round(split * float(data_size))))
        diff = splits_index[-1] - data_size
        for index in range(1, len(splits_index)):
            splits_index[index] -= diff
        assert splits_index[-1] == data_size

        shuffle_idx = get_shuffle_idx(seed, data_size)
        for split_i in range(len(splits)):
            shuffle_idx_split_file_name = f"{output_path}/{dataset_name}_seed{seed}_{split_name}_{data_split}_{split_i}.npy"
            shuffle_idx_split = shuffle_idx[
                splits_index[split_i]:splits_index[split_i + 1]]
            np.save(shuffle_idx_split_file_name,
                    shuffle_idx_split,
                    allow_pickle=True)
    index = np.load(index_file_name, allow_pickle=True)
    return index.tolist()


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset,
                 pad_token_id, train_phase) -> None:
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"],self.prompt_dataset[idx]["attention_mask"], \
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
def create_dataset_split(current_dataset, raw_dataset, train_phase, tokenizer,
                         end_of_conversation_token, max_seq_len):
    #将根据不同的阶段（train_phase）对数据集进行处理，主要是调用原先在PromptRawDataset类中定义的实例函数来实现。
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        # 因为phase1只需要用到chosen数据，所以只取chosen进行处理
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            # 获取chosen_sentence，即是将prompt和chosen拼接起来形成完整对话
            # 具体样例可参照“数据格式基本概念”中的样例
            chosen_sentence = raw_dataset.get_prompt_and_chosen(
                tmp_data)  # the accept response
            if chosen_sentence is not None:
                # 在对话末尾加入对话终止符
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

    elif train_phase == 2:
        # phase2需要用到chosen_sentence和reject_sentence
        # 所以需要对两者都进行处理
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

    elif train_phase == 3:
        # phase3用到prompt，prompt将被用来生成经验数据
        for i, tmp_data in enumerate(current_dataset):
            # tokenize the text
            # 直接获取prompt
            # 具体样例可参照“数据格式基本概念”中的样例
            prompt = raw_dataset.get_prompt(tmp_data)
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
                        y = prompt_token[key_word].squeeze(0)[length -
                                                              (max_seq_len -
                                                               1):].flip(0)
                    else:
                        # 将token进行flip（翻转/倒序），之后在data_collator中再将其flip回来
                        y = prompt_token[key_word].squeeze(0).flip(0)
                    prompt_token[key_word] = y
                prompt_dataset.append(prompt_token)
    # 返回PromptDataset实例，该实例相当于torch中的Dataset，可供DataLoader调用
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset,
                         tokenizer.pad_token_id, train_phase)


def create_dataset(local_rank, dataset_name, data_split, output_path,
                   train_phase, seed, tokenizer, end_of_conversation_token,
                   max_seq_len):
    raw_dataset = get_raw_dataset(dataset_name, output_path, seed, local_rank)
    train_dataset = raw_dataset.get_train_data()
    train_index = get_raw_dataset_split_index(local_rank, output_path,
                                              raw_dataset.dataset_name_clean,
                                              seed, "train", data_split,
                                              train_phase - 1,
                                              len(train_dataset))
    train_dataset = Subset(train_dataset, train_index)
    train_dataset = create_dataset_split(train_dataset, raw_dataset,
                                         train_phase, tokenizer,
                                         end_of_conversation_token,
                                         max_seq_len)

    eval_dataset = raw_dataset.get_eval_data()
    eval_index = get_raw_dataset_split_index(local_rank, output_path,
                                             raw_dataset.dataset_name_clean,
                                             seed, "eval",
                                             data_split, train_phase - 1,
                                             len(eval_dataset))
    eval_dataset = Subset(eval_dataset, eval_index)
    eval_dataset = create_dataset_split(eval_dataset, raw_dataset, train_phase,
                                        tokenizer, end_of_conversation_token,
                                        max_seq_len)
    return train_dataset, eval_dataset


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
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    fname = "_".join(data_path)
    sft_cache_key = "_".join(sft_only_data_path)
    tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    fname = "_".join(fname.split("/"))
    fname = hashlib.sha256(fname.encode()).hexdigest(
    )  # hash the file name to avoid too long file name
    train_fname = f"{output_path}/traindata_{fname}.pt"
    eval_fname = f"{output_path}/evaldata_{fname}.pt"

    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        if len(data_path) == 1:  # Single dataset.
            train_dataset, eval_dataset = create_dataset(
                local_rank, data_path[0], data_split, output_path, train_phase,
                seed, tokenizer, end_of_conversation_token, max_seq_len)
        else:  # Blending datasets.
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
            train_dataset = ConcatDataset(train_datasets)
            shuffle_idx = get_shuffle_idx(seed, train_size)
            train_dataset = Subset(train_dataset, shuffle_idx.tolist())
            eval_dataset = ConcatDataset(eval_datasets)
            shuffle_idx = get_shuffle_idx(seed, eval_size)
            eval_dataset = Subset(eval_dataset, shuffle_idx.tolist())

        # Append the SFT-only dataset if it exists, and current phase is 1(SFT).
        if train_phase == 1 and sft_only_data_path:
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
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
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
class DataCollatorReward:

    def __call__(self, data):
        """
                对dataloader取到的数据 data 进一步整理，将数据整理成batch输入形式
                入参 data 的具体样式可见下个代码块
        """
        batch = {}
        """f为data中的1个tuple，tuple的第0个元素和第2个元素
                分别为chosen_sentence和reject_sentence的input_ids
        """
        batch["input_ids"] = torch.cat([f[0]
                                        for f in data] + [f[2] for f in data],
                                       dim=0)
        """f为data中的1个tuple，tuple的第1个元素和第3个元素
                分别为chosen_sentence和reject_sentence的attention_mask
        """
        batch["attention_mask"] = torch.cat([f[1] for f in data] +
                                            [f[3] for f in data],
                                            dim=0)
        """batch的具体样式可见下个代码块"""
        return batch


class DataCollatorRLHF:

    def __init__(self, max_token_len, inference_tp_size):
        self.max_token_len = max_token_len
        self.inference_tp_size = inference_tp_size

    def __call__(self, data):
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
        return batch


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
最开始的时候载入过一次Dataset（见3.3.1），但刚开始载入的Dataset针对的是全部训练数据的管理，而此时使用的MiniDataset主要针对PPO训练迭代所使用的数据进行管理。PPO训练前的数据管理流程可以理解为：

Dataloader从Dataset中取出1个prompt_batch的无监督数据和1个prompt_batch的prompt数据；
使用1个prompt_batch的prompt数据进行经验采集，将得到1个prompt_batch的经验数据；
1个prompt_batch的无监督数据、1个prompt_batch的经验数据将被送入各自的MiniDataset实例进行管理：1个prompt_batch将被分成数个ppo_batch，供PPO训练进行数次迭代。
上述第3步就是MiniDataset所要做的事，其函数方法分别执行了：

add()：获取batch（prompt_batch）数据；
seperate()：细分为ppo_batch数据；
free():清空获取到的batch数据并返回ppo_batch数据。

'''
class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        '''
        :param max_size: batch数。通常此处指“用于给actor做生成的prompt的batch数（注意是batch数不是batch_size）”。
        :param small_batch_size: batch size。通常此处指“PPO训练的batch_size”。

        '''
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        #维护1个small_dataset
        small_dataset = []

        # 从self.dataset中逐个取batch
        for large_batch in self.dataset:
            # 判断batch的数据类型（列表 / 元组 / 字典），根据数据类型取其batch_size，赋值给large_size
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)

            '''
            以下部分代码略微抽象，需要举例说明
            - 比如prompt的batch_size设置为3，PPO训练用的batch_size设置为4，则最后能取来用、存入small_dataset的也就只有3条数据，因为生成用的dataloader只采样出了3条，最多也就只有3条。
            - 比如prompt的batch_size设置为5，PPO训练用的batch_size设置为4，则最后能取来用、存入small_dataset的就是2组数据（第1组为idx0,idx1,idx2,idx3共4条数据、第2组为idx4共1条数据）。
            - 比如prompt的batch_size设置为9，PPO训练用的batch_size设置为4，则最后能取来用、存入small_dataset的就是3组数据（[0,1,2,3],[4,5,6,7],[8]）。
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
