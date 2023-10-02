# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn
'''
2.3.3 RM的正向传播及成对排序损失
RM的正向传播过程不算复杂，总的来说就是：

数据经过主干网络得到shape为(bs*2, max_seq_len, hidden_size)的最后层输出特征hidden_states；
然后将输出特征送入线性层v_head得到shape为(bs*2, max_seq_len)的评分rewards。
较为复杂的部分实际上是“成对排序损失的计算”以及“评分聚合设计”。

成对排序损失（Pairwise Ranking Loss）
,,,,,,,,,,,,,,,,,


DeepSpeed-Chat在实现这部分时...........
分别选择了chosen_sentence和reject_sentence两者answer的对齐部分，通过文字叙述略显抽象，查看下方的代码块有助于你理解这个概念：

max_seq_len为10，pad_token_id为0，
有同属同个prompt的chosen_sentence和reject_sentence:
prompt: [11, 22, 33]
chosen_sentence: [11, 22, 33, 44, 55, 66, 0, 0, 0, 0]
reject_sentence: [11, 22, 33, 40, 50, 0, 0, 0, 0, 0]

“两者answer的对齐部分”即为“非prompt部分也非padding部分、但长度要对齐”：
chosen_truncated: [44, 55, 66]
reject_truncated: [40, 50, 0]

chosen_sentence的answer比较长，所以reject_sentence在取相应部分时要取至与chosen部分等长为止；
reject_sentence的answer较长时同理。
为了取到上述提及的“对齐部分”，代码进行了较为晦涩抽象的取index操作，但只要理解其最终目的是为了取到chosen_sentence和reject_sentence对齐部分的reward，来进行损失计算即可。

对话奖励设计
尽管使用的是“对齐部分”的reward来计算成对排序损失，但RM模型对一个对话的预测评分实际上取的是该对话文本最后一个有效token（通常会是“结束标记”）的reward，
下方代码块提供了一个简单例子说明了这个情况。

pad_token_id = 0
conversation = [11, 22, 33, 44, 55, 66, 0, 0, 0, 0]
conversation_rewards = [2.01, 0.23, 2.89, 0.66, 0.33, 2.25, 0.36, 0.99, 1.32, 1.62]
token_id为66的token作为该对话的最后1个有效token，
其对应的reward“2.25”将被用于表示整个对话的reward。




3.3.3.3 奖励reward_score和价值估计values的获取
奖励模型的模型类RewardModel中实现了相应的方法forward_value()，可支持输入“一句对话”返回“环境奖励与价值估计”。
与原先训练所用的方法forward()不同，forward()可支持输入“chosen-reject对话对”，主要实现了“对话对”之间排序损失的计算（forward()在【中篇】的2.3.3中已有所介绍，此处将不再赘述）。
以下通过简单例子来对“奖励”以及“价值估计”作区分：

“奖励/环境奖励/reward_score”主要是为对话序列给出一个奖励值/做出评分，
“价值估计/values”是为对话序列中的每一个位置都给出价值预测，是与时间步/状态紧密相关的。

有对话序列 seq=[11, 22, 33, 44, 55, 66, 0, 0, 0, 0]
其奖励reward_score只会是1个标量，如reward_score_seq=2.25；
其价值估计values是1维数组，如[2.01, 0.23, 2.89, 0.66, 0.33, 2.25, 0.36, 0.99, 1.32, 1.62]


'''
## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class RewardModel(nn.Module):
    """
    将读取得到的model的结构修改为适用于RewardModel的形式，
    总的来说即是使用载入的主干网络进行特征提取，
    其所提取的特征（最后层的各位置输出特征hidden_states）将被传入线性层，输出得到1个数值，
    该数值即为分值，因此max_seq_len维度的每个位置均会得到1个分值

    RM的模型结构基本如下所示（此处的基座模型为“facebook/opt-125m”），
    由主干网络rwtransformer及输出头v_head组成：

    RewardModel(
      (v_head): Linear(in_features=768, out_features=1, bias=False)
      (rwtranrsformer): OPTModel(
        (decoder): OPTDecoder(
          (embed_tokens): Embedding(50272, 768, padding_idx=1)
          (embed_positions): OPTLearnedPositionalEmbedding(2050, 768)
          (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (layers): ModuleList(
            (0-11): 12 x OPTDecoderLayer(
              (self_attn): OPTAttention(
                (k_proj): Linear(in_features=768, out_features=768, bias=True)
                (v_proj): Linear(in_features=768, out_features=768, bias=True)
                (q_proj): Linear(in_features=768, out_features=768, bias=True)
                (out_proj): Linear(in_features=768, out_features=768, bias=True)
              )
              (activation_fn): ReLU()
              (self_attn_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
              (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            )
          )
        )
      )
    )

    """
    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            """
            			OPT系列模型的word_embed_proj_dim为embedding层的输出维度，
            			通常在transformer模型中也就等于 hidden_size，
            			v_head将基于主干网络的输出特征 hidden_state 进行分值预测，共输出max_seq_len个分值
            """
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        """base_model即为主干网络，因此RM最终由1个主干网络和1个线性层构成"""
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
        """获得主干网络的输出的特征"""
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
        # 取最后一层的输出特征
        # hidden_states.shape: (bs * 2, max_seq_len, hidden_size)
        hidden_states = transformer_outputs[0]

        # v_head为Dx1的全连接网络对最后一维压缩
        # 将特征送入全连接层得到分数回归值
        # rewards.shape: (bs * 2, max_seq_len)
        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2

        # 先前提及过，实际的bs应该是输入bs的一半
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        # 区分出chosen和reject
        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        for i in range(bs):
            # 取出同组chosen和rejected的token_id和分值reward
            # chosen_id.shape: (max_seq_len,)
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]
            """
            下方本应有各种取index相关的操作，
            基于源码解读的可读性考量，且这些部分只是逻辑形式上的弯弯绕绕，与相关原理并不存在直接关系，
            所以我选择暂且将它们忽略。

            c_ind为chosen_sentence的answer后的第一个pad_token的index
            例如pad_token_id=0，sentence[11,22,33,44,55,66,0,0,0,0]，c_ind即为第一个pad_token的index=6。
            
            r_ind同理，为reject_sentence的answer后的第一个pad_token的index
            
            end_ind则为两者的较大者
            
            取chosen和rejected第一个不同的地方的index，可以理解为“response中两个回答自由发挥的第1个token的index”
            divergence_ind为chosen_sentence和reject_sentence两者answer的第1个token的index

            """

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

            # 以chosen_sentence和reject_sentence最先不同的地方为起始、生成结束的地方为终止，取两者在这个片段的对应分值
            # 这部分其实就是上个代码块提及的“对齐部分”
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]

            # 取代表结束的pad token所在位置的前一个位置（可以理解为的最后一个有效token的位置）的分值作为参考分值
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])
            #pair wise loss代码如下，如果给pair里边好的句子打分高（c_truncated_reward），坏的句子（r_truncated_reward）打分低，loss就会小：
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)

        # 返回损失和参考分值
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
        """
                和forward有些差别，forward需要针对输入的chosen-rejected对计算排序损失并返回
                而forward_value只需要考虑一个输入，然后返回分值
                :param return_value_only: 如果设置为True，则在计算出values（在序列上每个位置的分值预测）后直接返回
        """

        # 经过主干网络正向传播得到输出
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)

        # hidden_states.shape: (bs, max_seq_len, hidden_size)
        hidden_states = transformer_outputs[0]

        # 将隐状态特征传入线性层v_head输出得到分值"""
        # values.shape: (bs, max_seq_len)
        values = self.v_head(hidden_states).squeeze(-1)
        if return_value_only:
            """
               如果传参中预设了“return_value_only=True”，
               那么将直接返回 values: (bs, max_seq_len)
            """
            return values
        else:
            # 否则还将进一步取得reward_score
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            for i in range(bs):
                input_id = input_ids[i]

                # value.shape: (max_seq_len,)
                value = values[i]

                # c_ind即为prompt之后的序列片段中，第一个pad_token的index
                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()

                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len

                #取c_ind的前一个index（实际上就是answer的最终位置）作为reward_score
                chosen_end_scores.append(value[c_ind - 1])

            #返回values和reward_score"""
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
