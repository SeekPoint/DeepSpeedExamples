# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset
from torch.utils.data import Subset
import re
from pydebug import gd, infoTensor

# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
'''
0.2.3.1 自定义PromptRawDataset类
UML时序图(3-6)
...

自定义的数据集可以继承自上述的“PromptRawDataset”类，
例如class CustomDataset(PromptRawDataset)，
然后重写其中的self.dataset_name及self.dataset_clean_name，
此处的“dataset_name”即为传参指定数据集时所要填写的名称，
例如self.dataset_name=custom，在设置传参--data_path=‘custom’时，
将会读取到CustomDataset的数据用于进行训练。
另外其中的get_train_data()等实例函数也需要进行重写，
主要是实现将原始数据处理成注释所提及格式。

'''

# create_prompt_dataset解析
# create_prompt_dataset这个函数实际上直接或者间接的用到了utils/data中raw_dataset.py和data_utils.py，
# 为了搞清楚这个函数，我们需要对这两个文件做一个解析。
#
# 首先解析一下raw_dataset.py。这里先定义了一个PromptRawDataset类：

# 这段代码定义了一个名为PromptRawDataset的类，这个类是一个模板类，用于处理和组织模型输入数据的格式。
# 如果有新的数据集需要进行处理，可以继承这个类并实现相应的方法来确保数据的统一格式和接口。
class PromptRawDataset(object):
    # 首先，这个类的构造函数__init__接收四个参数：output_path（输出路径），seed（随机种子），
    # local_rank（本地等级）和dataset_name（数据集名称）。
    # 在构造函数中，如果数据集名称不是'local/jsonfile'，
    # 那么会使用Hugging Face的datasets库的load_dataset函数来加载数据集。
    # 该类是一个模板，它定义了一套统一的API和数据格式，所有新的数据集都需要按照这个模板来进行适配。
    def __init__(self, output_path, seed, local_rank, dataset_name):
        """
        初始化
        :param output_path: 输出缓存路径。
        :param seed: 随机种子。
        :param local_rank: 当前进程序号。
        :param dataset_name: 数据集名称，后续指定所需读取的数据集时将以名称为准。
        """
        self.output_path = output_path # 数据集存储的路径
        self.seed = seed # 随机种子
        self.local_rank = local_rank # 用于分布式训练中确定当前进程使用哪部分数据
        gd.debuginfo(prj="ds_chat", info=f"C:{self.__class__.__name__}")
        gd.debuginfo(prj="ds_chat", info=f'dataset_name={dataset_name}')

        if not dataset_name == 'local/jsonfile':
            gd.debuginfo(prj="ds_chat", info=f" not  local/jsonfile ")
            # 加载数据集
			
            # load_dataset源自datasets库，该方法支持读取csv/json/text等多种文件格式的数据
            '''
            https://stackoverflow.com/questions/77020278/how-to-load-a-huggingface-dataset-from-local-path
                        
            ll ~/hf_model/rm-static/data
            total 71312
            drwxrwxr-x 2 amd00 amd00     4096 10月  7 21:46 ./
            drwxrwxr-x 4 amd00 amd00     4096 10月  7 21:46 ../
            -rw-rw-r-- 1 amd00 amd00  4609580 10月  7 21:14 test-00000-of-00001-8c7c51afc6d45980.parquet
            -rw-rw-r-- 1 amd00 amd00 68396955 10月  7 21:15 train-00000-of-00001-2a1df75c6bce91ab.parquet
            
            '''
            # data_files = {"train":"train-00000-of-00001-2a1df75c6bce91ab.parquet",
            #               "test": "test-00000-of-00001-8c7c51afc6d45980.parquet"}
            #
            # self.raw_datasets = load_dataset("parquet", data_dir ='~/s_data/hf_model/rm-static/data/',
            #                         data_files = data_files)

            s_data_files = {"train":"train-small.parquet",
                          "test": "test-small.parquet"}

            self.raw_datasets = load_dataset("parquet", data_dir ='~/s_data/hf_model/rm-static/data/',
                                    data_files = s_data_files)
            gd.debuginfo(prj="ds_chat", info=f"self.raw_datasets is: {self.raw_datasets}")


    # 然后，这个类定义了一些方法，这些方法在默认情况下并没有实现（只是返回None或者空操作），
    # 这是因为这个类只是一个模板类，这些方法需要在实际使用时在子类中具体实现。
    def get_train_data(self): # 获取训练数据
        """
        获取训练集
        :return: dataset数据格式
        """
        return

    def get_eval_data(self):  # 获取评估数据
        """
        获取验证集
        :return: dataset数据格式
        """
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    # get_prompt方法用于获取样本中的prompt（提示，这是模型的输入）。
    def get_prompt(self, sample):
        """
        从dataset的sample（单个样本）中获取prompt。
        :param sample: dataset的元素
        :return: prompt。prompt的格式必须为 "Human: {} Assistant:".format(actual_prompt_sentence)
        """
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    # get_chosen方法用于获取样本中的chosen（已选的回应，这是模型需要生成的目标输出）。
    def get_chosen(self, sample):
        """
        从dataset的sample（单个样本）中获取chosen。chosen实际上是“chosen response”，
        指的是“精选的回复”，即人类所偏好的、高分的回复。
        :param sample: dataset的元素
        :return: chosen。chosen的格式必须为" {}".format(actual_response_sentence)
        """
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    # get_rejected方法用于获取样本中的rejected（被拒绝的回应，这可能用于一些特定的训练场景，
    # 比如在对抗训练中，但如果数据集中没有这样的数据，可以返回None）。
    def get_rejected(self, sample):
        """
        从dataset的sample（单个样本）中获取rejected。rejected实际上是“rejected response”，
        指的是“排斥的回复”，即人类所厌恶的、低分的回复。
        :param sample: dataset的元素
        :return: rejected。如果数据集中不存在则返回为None；
        如果存在，则其格式必须为 " {}".format(actual_response_sentence)
        """
        return

    # 获取样本中的prompt和chosen
    def get_prompt_and_chosen(self, sample):
        """
        从dataset的sample（单个样本）中获取prompt与chosen。
        :param sample: dataset的元素
        :return: prompt与chosen的衔接。同样需要满足上述格式要求，即衔接结果为
        "Human: {} Assistant: {}".format(actual_prompt_sentence, actual_response_sentence)
        """
        return

    # 获取样本中的prompt和rejected
    def get_prompt_and_rejected(self, sample):
        """
        从dataset的sample（单个样本）中获取prompt与rejected。
        :param sample: dataset的元素
        :return: prompt与rejected的衔接。同样需要满足上述格式要求，即衔接结果为
        "Human: {} Assistant: {}".format(actual_prompt_sentence, actual_response_sentence)
        """
        return

# English dataset
class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/rm-static" # 数据集名称
        self.dataset_name_clean = "Dahoas_rm_static"  # 数据集名称
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        # 返回训练数据
        return self.raw_datasets["train"]

    def get_eval_data(self):
        # 返回验证数据
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        # 从样本中提取prompt，也就是输入的初始提示。
        return sample['prompt']

    def get_chosen(self, sample):
        # 从样本中提取chosen，可能表示由模型选中或者用户接受的回复。
        return sample['chosen']

    def get_rejected(self, sample):
        # 从样本中提取rejected，可能表示被模型拒绝或者用户不接受的回复。
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        # 从样本中提取prompt和chosen，并将它们拼接在一起。
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        # 从样本中提取prompt和rejected，并将它们拼接在一起。
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    # 只获取prompt字段的数据
    def get_prompt(self, sample):
        return " Human: " + sample['prompt'] + " Assistant:"

    # The chosen response should be in the format of: " " + actual_response_sentence
    # 只获取chosen字段的数据
    def get_chosen(self, sample):
        return " " + sample['chosen']

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    # 只获取rejected字段的数据
    def get_rejected(self, sample):
        return " " + sample['rejected']

    # 同时获取prompt和chosen的数据
    # 这两段数据一一拼接后可以训练SFT
    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample['chosen']

    # 同时获取prompt和rejected数据
    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample[
            'rejected']


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt'] + "Assistant:"

    def get_chosen(self, sample):
        return sample['chosen'].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample['rejected'].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

# 接下来就是每个具体数据集的定义，我这里以 OpenaiWebgptcomparisonsDataset 为例解析一下，剩下的读者又需要可以自行理解：

# English dataset
# 这个类OpenaiWebgptcomparisonsDataset继承自PromptRawDataset类，
# 针对"openai/webgpt_comparisons"这个具体的数据集进行了特化。
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):
    # 在构造函数__init__中，调用了父类的构造函数，并设定了dataset_name和dataset_name_clean两个属性，
    # 分别为"openai/webgpt_comparisons"和"openai_webgpt_comparisons"。
    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    # get_train_data和get_eval_data方法分别从raw_datasets中获取训练数据和测试数据。
    # 它们与之前的DahoasRmstaticDataset类不同之处在于，它们使用get_raw_dataset_split_index
    # 方法对训练数据进行了划分，将其划分为训练集和验证集，并返回对应的数据子集。
    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    # get_prompt，get_chosen和get_rejected方法分别从样本中获取提示，已选回应和被拒绝的回应。
    # 这里假定样本是一个字典，其中包含了名为'question'，'score_0'，'score_1'，'answer_0'和'answer_1'的字段。
    # 其中，'question'字段是一个字典，包含了'full_text'字段。这个字段包含了人类提出的问题。
    # 'score_0'和'score_1'字段是字符串，表示对'answer_0'和'answer_1'的评分。
    # 如果'score_0'大于等于'score_1'，那么'answer_0'就是已选回应，'answer_1'就是被拒绝的回应，反之亦然。
    # 在获取已选回应和被拒绝的回应时，还对回应进行了处理，
    # 去除了所有形如"[...]"或"(...)"的文本，然后在回应前添加了一个空格。
    def get_prompt(self, sample):
        return " Human: " + sample['question']['full_text'] + " Assistant:"

    def get_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    # get_prompt_and_chosen和get_prompt_and_rejected方法则分别返回样本中的'prompt'和'chosen'，
    # 以及'prompt'和'rejected'的组合。这两个方法的返回值可以直接作为模型的输入和目标输出。
    # 在返回这两个组合时，也进行了类似的处理，去除了所有形如"[...]"或"(...)"的文本。
    def get_prompt_and_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "stanfordnlp/SHP"
        self.dataset_name_clean = "stanfordnlp_SHP"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['history'] + " Assistant:"

    def get_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " " + response

    def get_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " Human: " + sample['history'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " Human: " + sample['history'] + " Assistant: " + response


# English dataset
class PvduySharegptalpacaoavicunaformatDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "pvduy/sharegpt_alpaca_oa_vicuna_format"
        self.dataset_name_clean = "pvduy_sharegpt_alpaca_oa_vicuna_format"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        if sample['prompt'] is not None and len(sample['prompt']) > 0:
            return sample['prompt'].replace("USER", "Human").replace(
                "ASSISTANT", "Assistant")
        return None

    def get_chosen(self, sample):
        if sample['label'] is not None and len(sample['label']) > 0:
            return " " + sample['label']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['label'] is not None and len(
                sample['prompt']) > 0 and len(sample['label']) > 0:
            return sample['prompt'].replace("USER", "Human").replace(
                "ASSISTANT", "Assistant") + " " + sample['label']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


class LocalJsonFileDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             chat_path + '/data/train.json',
                                             "eval":
                                             chat_path + '/data/eval.json'
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return " " + sample['prompt']
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        if sample['chosen'] is not None:
            return " " + sample['chosen']
        return None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        if sample['rejected'] is not None:
            return " " + sample['rejected']
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['chosen'] is not None:
            return " " + sample['prompt'] + " " + sample['chosen']
        return None

    def get_prompt_and_rejected(self, sample):
        if sample['prompt'] is not None and sample['rejected'] is not None:
            return " " + sample['prompt'] + " " + sample['rejected']
        return None


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "wangrui6/Zhihu-KOL"
        self.dataset_name_clean = "wangrui6_Zhihu_KOL"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['INSTRUCTION'] is not None:
            return " Human: " + sample['INSTRUCTION'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['RESPONSE'] is not None:
            return " " + sample['RESPONSE']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['INSTRUCTION'] is not None and sample['RESPONSE'] is not None:
            return " Human: " + sample[
                'INSTRUCTION'] + " Assistant: " + sample['RESPONSE']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Cohere/miracl-zh-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_zh_queries_22_12"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'negative_passages'][0]['text']


# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Hello-SimpleAI/HC3-Chinese"
        self.dataset_name_clean = "Hello_SimpleAI_HC3_Chinese"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['question'] is not None:
            return " Human: " + sample['question'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['human_answers'][0] is not None:
            return " " + sample['human_answers'][0]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['question'] is not None and sample['human_answers'][
                0] is not None:
            return " Human: " + sample['question'] + " Assistant: " + sample[
                'human_answers'][0]
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mkqa-Chinese"
        self.dataset_name_clean = "mkqa"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['zh_cn'] is not None:
            return " Human: " + sample['queries']['zh_cn'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['zh_cn'][0]['text'] is not None:
            return " " + sample['answers']['zh_cn'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['zh_cn'] is not None and sample['answers'][
                'zh_cn'][0]['text'] is not None:
            return " Human: " + sample['queries'][
                'zh_cn'] + " Assistant: " + sample['answers']['zh_cn'][0][
                    'text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mkqa-Japanese"
        self.dataset_name_clean = "mkqa"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['ja'] is not None:
            return " Human: " + sample['queries']['ja'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['ja'][0]['text'] is not None:
            return " " + sample['answers']['ja'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['ja'] is not None and sample['answers']['ja'][0][
                'text'] is not None:
            return " Human: " + sample['queries'][
                'ja'] + " Assistant: " + sample['answers']['ja'][0]['text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Cohere/miracl-ja-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_ja_queries_22_12"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        if len(sample['negative_passages']) > 0:
            return " Human: " + sample['query'] + " Assistant: " + sample[
                'negative_passages'][0]['text']
        return None


# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmqg/qg_jaquad"
        self.dataset_name_clean = "lmqg_qg_jaquad"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['question'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['sentence']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['question'] + " Assistant: " + sample[
            'sentence']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmqg/qag_jaquad"
        self.dataset_name_clean = "lmqg_qag_jaquad"
        gd.debuginfo(prj="ds_chat", info=self.__class__.__name__ + '__' + self.dataset_name_clean)

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['paragraph']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant: " + sample[
            'paragraph']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None
