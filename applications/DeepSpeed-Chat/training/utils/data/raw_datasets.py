# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from datasets import load_dataset
from torch.utils.data import Subset
import re


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
'''
自定义的数据集可以继承自上述的“PromptRawDataset”类，
例如class CustomDataset(PromptRawDataset)，
然后重写其中的self.dataset_name及self.dataset_clean_name，
此处的“dataset_name”即为传参指定数据集时所要填写的名称，
例如self.dataset_name=custom，在设置传参--data_path=‘custom’时，
将会读取到CustomDataset的数据用于进行训练。
另外其中的get_train_data()等实例函数也需要进行重写，
主要是实现将原始数据处理成注释所提及格式。

'''
class PromptRawDataset(object):
    def __init__(self, output_path, seed, local_rank, dataset_name):
        """
        初始化
        :param output_path: 输出缓存路径。
        :param seed: 随机种子。
        :param local_rank: 当前进程序号。
        :param dataset_name: 数据集名称，后续指定所需读取的数据集时将以名称为准。
        """
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if not dataset_name == 'local/jsonfile':
            # load_dataset源自datasets库，该方法支持读取csv/json/text等多种文件格式的数据
            self.raw_datasets = load_dataset(dataset_name)

    def get_train_data(self):
        """
              获取训练集
              :return: dataset数据格式
        """
        return

    def get_eval_data(self):
        """
                获取验证集
                :return: dataset数据格式
        """
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        """
                从dataset的sample（单个样本）中获取prompt。
                :param sample: dataset的元素
                :return: prompt。prompt的格式必须为 "Human: {} Assistant:".format(actual_prompt_sentence)
        """
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        """
                从dataset的sample（单个样本）中获取chosen。chosen实际上是“chosen response”，指的是“精选的回复”，即人类所偏好的、高分的回复。
                :param sample: dataset的元素
                :return: chosen。chosen的格式必须为" {}".format(actual_response_sentence)
    """
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        """
                从dataset的sample（单个样本）中获取rejected。rejected实际上是“rejected response”，指的是“排斥的回复”，即人类所厌恶的、低分的回复。
                :param sample: dataset的元素
                :return: rejected。如果数据集中不存在则返回为None；如果存在，则其格式必须为 " {}".format(actual_response_sentence)
        """
        return

    def get_prompt_and_chosen(self, sample):
        """
                从dataset的sample（单个样本）中获取prompt与chosen。
                :param sample: dataset的元素
                :return: prompt与chosen的衔接。同样需要满足上述格式要求，即衔接结果为
                "Human: {} Assistant: {}".format(actual_prompt_sentence, actual_response_sentence)
        """
        return

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
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"

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
class DahoasFullhhrlhfDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"

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
        return " Human: " + sample['prompt'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['chosen']

    def get_rejected(self, sample):
        return " " + sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample[
            'rejected']


# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"

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


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"

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
