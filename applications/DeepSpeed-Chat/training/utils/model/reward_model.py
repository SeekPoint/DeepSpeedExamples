# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn

## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)

        '''
        https://zhuanlan.zhihu.com/p/624589622
        详解大模型RLHF过程（配代码解读）
        
        奖励（reward）模型训练
        首先要声明的是，在强化学习阶段，用到的reward model和critic model都使用同一个模型初始化，
        因此在训练reward模型的过程中，也是在训练critic model。
        其次对符号进行说明，大模型中间隐藏层的参数维度为(B,L,D)，B为batch size大小，L为句子长度，D为embedding维度。
        在接下来的代码讲解中，我也会标明代码中各个变量的维度，以更好的理解其意义。
        
        在进行RLHF时，需要一个奖励模型来评估语言大模型（actor model）回答的是好是坏，
        这个奖励模型通常比被评估的语言大模型小一些（deepspeed的示例中，语言大模型66B，奖励模型只有350M）。
        奖励模型的输入是prompt+answer的形式，让模型学会对prompt+answer进行打分。
        奖励模型最后一层隐藏层的输出维度为(B,L,D)，通过一个D✖️1的全连接层将维度变为(B, L)，在L这个维度上，
        第i个位置的数据表示：从第i个位置到最后一个位置输出所能获得的奖励分值的累加和（和DQN里边的Q值一个意义），这种形式的输出满足了critic model的输出要求。
        对应代码如下：
        '''

        # huggingface模型返回值是个list，第0位是模型最后输出的hideen state
        hidden_states = transformer_outputs[0]

        # v_head为Dx1的全连接网络对最后一维压缩
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            '''
            对于一个奖励模型来说，目标是给一个句子进行打分，按理说每个句子对应一个分值就行了，但是目前对于长度为L的句子，奖励模型输出了L个值。
            我们用L维度上的最后一个位置的值当作为本句话的奖励得分。
            奖励模型训练优化采用pair wiss loss，即同时输入模型关于同一个问题的两个回答，让模型学会这两个句子哪个分高哪个分低。
            之所以如此训练是因为，在给奖励模型进行数据标注的过程中，给同一个问题的不同回答量化的打具体分值比较难，但是对他们进行排序相对简单，
            代码如下：
            '''
            # 同一个batch里边的句子需要等长，短句后边会被padding
            # [divergence_ind:end_ind]索引了padding前一个位置的输出分值
            # chosen_reward是同一个句子pair里分数高的句子，r_truncated_reward是句子pair里分数低的句子
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])
            #pair wise loss代码如下，如果给pair里边好的句子打分高（c_truncated_reward），坏的句子（r_truncated_reward）打分低，loss就会小：
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }
    '''
    在训练强化学习的过程中，会用到reward model（critic model，再次提醒，critic model和reward model是同一个模型的两个副本）的推理过程，
    通过调用forward_value实现，具体代码如下，返回的值中有两种值，values表示每个位置i，从第i个位置到最后一个位置的奖励累加值，
    供强化学习过程中critic model使用；“chosen_end_scores”指的是对每个prompt+answer的打分，供reward model使用。
    '''
    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
