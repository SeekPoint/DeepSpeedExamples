# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn
from pydebug import debuginfo, infoTensor

'''
3.3.3.3 奖励reward_score和价值估计values的获取
奖励模型的模型类RewardModel中实现了相应的方法forward_value()，可支持输入“一句对话”返回“环境奖励与价值估计”。
与原先训练所用的方法forward()不同，forward()可支持输入“chosen-reject对话对”，
主要实现了“对话对”之间排序损失的计算（forward()在【中篇】的2.3.3中已有所介绍，此处将不再赘述）。
以下通过简单例子来对“奖励”以及“价值估计”作区分：

    “奖励/环境奖励/reward_score”主要是为对话序列给出一个奖励值/做出评分，
    “价值估计/values”是为对话序列中的每一个位置都给出价值预测，是与时间步/状态紧密相关的。
    
    有对话序列 seq=[11, 22, 33, 44, 55, 66, 0, 0, 0, 0]
    其奖励reward_score只会是1个标量，如reward_score_seq=2.25；
    其价值估计values是1维数组，如[2.01, 0.23, 2.89, 0.66, 0.33, 2.25, 0.36, 0.99, 1.32, 1.62]


'''
'''
3.3 定义Reward模型：
定义reward模型：选择OPT-350M模型作为backbone，并定义一个linear层用于分类。

    OPT模型中，需要定义–num_padding_at_beginning=1，OPT默认首个字符为PAD token;
    
    对于每个chosen或rejected tokens，取第一个padding token的前一个token的得分作为当前chosen或rejected input的得分
    
    For RW, the training objective is the pairwise ranking score, i.e., for the two query-answer pairs, 
    RM is supposed to give a higher score to the better answer. There are multiple ways to achieve this.
    In our implementation, 
    we use either the end token of the sequence or the first padding token as the aggregated score and compare them. 
    Others may also use the average score for the entire answer as an alternative.

Reward函数细节详见代码和注释：
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
        ## 为当前输入打分, 每个token都会对应一个分数

        super().__init__()

        # 基础模型的配置信息
        self.config = base_model.config

        # 序列开始处的填充数量
        self.num_padding_at_beginning = num_padding_at_beginning

        # 检查配置是否包含word_embed_proj_dim属性
        if hasattr(self.config, "word_embed_proj_dim"):
            debuginfo(prj='ds-chat', info=self.__class__.__name__ + '-- word_embed_proj_dim')
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            """
            OPT系列模型的word_embed_proj_dim为embedding层的输出维度，
            通常在transformer模型中也就等于 hidden_size，
            v_head将基于主干网络的输出特征 hidden_state 进行分值预测，共输出max_seq_len个分值
            """
            # 使用一个线性层self.v_head将word_embed_proj_dim映射到1，这个线性层没有偏置
            self.v_head = nn.Linear(self.config.word_embed_proj_dim,
                                    1,
                                    bias=False)
        else:
            debuginfo(prj='ds-chat', info=self.__class__.__name__ + '--No word_embed_proj_dim')
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``

            # 检查配置是否包含hidden_size属性。如果包含，那么将其赋值给n_embd
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd

            # 使用一个线性层self.v_head将n_embd映射到1，这个线性层没有偏置。
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)

        # base_model即为主干网络，因此RM最终由1个主干网络和1个线性层构成
        self.rwtranrsformer = base_model

        # 填充token的ID
        self.PAD_ID = tokenizer.pad_token_id

    # 启用梯度检查点，减少内存使用
    def gradient_checkpointing_enable(self):
        debuginfo(prj='ds-chat', info=self.__class__.__name__)

        # 启用后，在反向传播过程中，rwtranrsformer会重新计算一些中间层的输出，
        # 而不是把这些输出在整个前向传播和反向传播过程中存储在内存中。
        self.rwtranrsformer.gradient_checkpointing_enable()

    # 禁用梯度检查点
    def gradient_checkpointing_disable(self):
        debuginfo(prj='ds-chat', info=self.__class__.__name__)

        # 禁用后，rwtranrsformer在反向传播过程中不再重新计算一些中间层的输出，
        # 而是把这些输出在整个前向传播和反向传播过程中存储在内存中。
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None, # 模型的输入数据
                past_key_values=None, # 这个参数用于Transformer模型中的decoder。
                                      # 在进行序列生成任务时，decoder可以使用之前时间步的隐藏状态（key和value）来加速计算。
                attention_mask=None, # 指定哪些元素不应该被注意力机制关注到
                position_ids=None, # 每个输入token的位置编码
                head_mask=None, # 用于屏蔽某些注意力头（attention head），也就是让模型在计算时忽略某些注意力头的输出。
                inputs_embeds=None, # 非None时，模型将使用此值作为输入而不是input_ids
                use_cache=False # 是否使用缓存来加速解码
        ):

        """
        假设默认设置的batch_size为N，那么len(input_ids)=2*N
        其中前N个为chosen input ids（正样本），后N个为rejected input ids（负样本）
        """
        loss = None

        # debuginfo(prj='ds-chat', info=self.__class__.__name__)

        """获得主干网络的输出的特征"""
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)

        print("transformer_outputs is-1", transformer_outputs)

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
        第i个位置的数据表示：从第i个位置到最后一个位置输出所能获得的奖励分值的累加和（和DQN里边的Q值一个意义），
        这种形式的输出满足了critic model的输出要求。
        对应代码如下：
        '''

        # huggingface模型返回值是个list，第0位是模型最后输出的hideen state
        # 取最后一层的输出特征
        # hidden_states.shape: (bs * 2, max_seq_len, hidden_size)
        # 是transformer模型的主要输出，表示模型对输入的隐藏表示。
        hidden_states = transformer_outputs[0]
        # print("hidden_states is:", hidden_states)

        # 将hidden_states传递到self.v_head（一个全连接层）并删除最后一个维度（通过squeeze(-1)）来计算reward。
        # 这个值可以被视为模型对每个输入token的奖励

        # v_head为Dx1的全连接网络对最后一维压缩
        # 将特征送入全连接层得到分数回归值
        # rewards.shape: (bs * 2, max_seq_len)
        # 全连接层将每个token的隐藏状态映射到一个实数，即模型对该token的奖励。
        rewards = self.v_head(hidden_states).squeeze(-1)
        # print("rewards is:", rewards)


        # print("T hidden_states:", infoTensor(hidden_states)) #only ph2
        # T hidden_states: _Size([16, 128, 768])_float16_cuda:1_
        # print("T rewards:", infoTensor(rewards)) #only ph2

        # 模型选择或拒绝某些特定输入的平均分数
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        # input_ids的形状是[batch_size, sequence_length]
        assert len(input_ids.shape) == 2

        # 批处理大小的一半
        # 先前提及过，实际的bs应该是输入bs的一半
        bs = input_ids.shape[0] // 2

        # 序列长度
        seq_len = input_ids.shape[1]

        # 区分出chosen和reject
        # N x seq x 1 前N个为正样本
        # 选择前半部分作为被选定的输入
        chosen_ids = input_ids[:bs]

        # 后N个为负样本
        # 选择后半部分作为被拒绝的输入
        rejected_ids = input_ids[bs:]

        # chosen 和 reject都会生成一个对应的分数 rewards

        # 获得前N个正样本的预测的reward
        # 前半部分选定的奖励
        chosen_rewards = rewards[:bs]

        # 获得后N个负样本的预测的reward
        # 后半部分被拒绝的奖励
        rejected_rewards = rewards[bs:]

        # print("len of chosen_ids is:", len(chosen_ids))
        # print("len of rejected_ids is:", len(rejected_ids))
        # print("len of chosen_rewards is:", len(chosen_rewards))
        # print("len of rejected_rewards is:", len(rejected_rewards))
        # print("bs is:", bs)
        # print("seq_len is:", seq_len)
        '''
        len of chosen_ids is: 8
        len of rejected_ids is: 8
        len of chosen_rewards is: 8
        len of rejected_rewards is: 8
        bs is: 8
        seq_len is: 128
        '''

        # 计算配对损失，应用于强化学习中，主要目的是比较生成模型生成的不同序列（被接受和被拒绝的序列）的奖励
        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0

        # 遍历每个样本
        for i in range(bs):
            # 遍历每个序列对（被接受和被拒绝的序列），提取相应的ID和奖励

            # 取出同组chosen和rejected的token_id和分值reward
            # chosen_id.shape: (max_seq_len,)
            # 获得一个chosen样本（正样本）
            chosen_id = chosen_ids[i]
            # print("chosen_id is:", chosen_id)

            # 获得一个rejected样本（负样本）
            rejected_id = rejected_ids[i]
            # print("rejected_id is:", rejected_id)

            # 当前正样本的得分
            chosen_reward = chosen_rewards[i]
            # print("chosen_reward is:", chosen_reward)

            # 当前负样本的得分
            rejected_reward = rejected_rewards[i]
            # print("rejected_reward is:", rejected_reward)


            # print("T chosen_id--5:", infoTensor(chosen_id))
            # print("T rejected_id--5:", infoTensor(rejected_id))
            # print("T chosen_reward--5:", infoTensor(chosen_reward))
            # print("T rejected_reward--5:", infoTensor(rejected_reward))
            '''
            only ph2
            T chosen_id--5: _Size([128])_int64_cuda:0_
            T rejected_id--5: _Size([128])_int64_cuda:0_
            T chosen_reward--5: _Size([128])_float16_cuda:0_
            T rejected_reward--5: _Size([128])_float16_cuda:0_
            '''

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

            ## 获取 PAD_token 的索引, 只支持right_padding ?

            # 找出在被选定序列中哪些位置存在填充（PAD）的标记，并选择第一个非填充标记的位置作为序列的结束位置。
            # 如果不存在填充标记，则使用整个序列的长度作为结束位置。
            # PyTorch 中的nonzero()方法，这个方法会返回输入张量中所有非零元素的索引
            # 获得所有padding token的索引
            c_inds = (chosen_id == self.PAD_ID).nonzero()
            # print("c_inds is:", c_inds)

            # 如果是OPT，那么第0个一定是OPT模型默认在input最前面的padding token，不予考虑
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len
            # OPT model pads the first token, so we need to use the second padding token as the end of the sequence

            # print("c_ind is:", c_ind)

            # 获取不同id的索引
            # 查找被选定序列和被拒绝序列之间的差异位置，这里的差异位置是指这两个序列第一个不同的标记的位置。
            check_divergence = (chosen_id != rejected_id).nonzero()  # [[0, 0], [1, 0], ..., [seq_len, 0]]
            # print("check_divergence is:", check_divergence)

            # 如果两个序列完全相同，则结束位置设置为序列的最后一位，差异位置设置为结束位置的前一位。
            # 说明不存在相等的padding token
            if len(check_divergence) == 0:
                # 获取序列的长度
                end_ind = rejected_reward.size(-1)

                # 原因：虽然两个序列没有差异，但我们仍然需要一个"差异点"，因为这个位置将被用于计算奖励，所以选择最后一个位置的前一位
                divergence_ind = end_ind - 1

                # 在这种没有差异的情况下，被选定的序列和被拒绝的序列应该有相同的长度
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                # 如果两个序列不完全相同，再次对被拒绝的序列执行相同的填充检查过程，并确定新的结束位置和差异位置。
				# 找出所有填充的位置
                r_inds = (rejected_id == self.PAD_ID).nonzero()

                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len

                # 确定结束位置end_ind，结束位置应该是两个序列中的最大值
                end_ind = max(c_ind, r_ind)

                # 确定差异位置
                divergence_ind = check_divergence[0]

            assert divergence_ind > 0
            '''
            对于一个奖励模型来说，目标是给一个句子进行打分，
            按理说每个句子对应一个分值就行了，但是目前对于长度为L的句子，奖励模型输出了L个值。
            我们用L维度上的最后一个位置的值当作为本句话的奖励得分。
            奖励模型训练优化采用pair wiss loss，即同时输入模型关于同一个问题的两个回答，让模型学会这两个句子哪个分高哪个分低。
            之所以如此训练是因为，在给奖励模型进行数据标注的过程中，
            给同一个问题的不同回答量化的打具体分值比较难，但是对他们进行排序相对简单，
            代码如下：
            '''
            # 同一个batch里边的句子需要等长，短句后边会被padding
            # [divergence_ind:end_ind]索引了padding前一个位置的输出分值
            # chosen_reward是同一个句子pair里分数高的句子，r_truncated_reward是句子pair里分数低的句子

            # 以chosen_sentence和reject_sentence最先不同的地方为起始、生成结束的地方为终止，取两者在这个片段的对应分值
            # 这部分其实就是上个代码块提及的“对齐部分”

            # 从不相同的第一个token到 最后一个token的reward
            # 基于差异位置和结束位置，截取被选定序列和被拒绝序列的奖励部分
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            # print("c_truncated_reward is:", c_truncated_reward)
            # print("r_truncated_reward is:", r_truncated_reward)

            # 从截取的奖励中计算平均值并添加到被选定和被拒绝的平均分列表中
            # 选择最后一个非填充token的奖励值添加到得分列表，可能是因为最后一个token往往包含了模型对整个输入序列的最终评价
            # 取代表结束的pad token所在位置的前一个位置（可以理解为的最后一个有效token的位置）的分值作为参考分值
            chosen_mean_scores.append(chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            # 使用logsigmoid函数计算两个截取奖励之差的对数值，并将结果添加到损失中
            # logsigmoid是一种将任何实数映射到（0，1）区间的函数
            # 当c_truncated_reward大于r_truncated_reward时，这个值会接近于1；
            # 否则，它会接近于0。这样可以使得损失更加稳定，避免在梯度下降时步长过大。
            # 最后一个token的分数为reward model 的reference 分数。训练critic，为一一对应的loss
            # chosen 的reward尽可能大，reject reward尽可能小
            # pair wise loss代码如下，
            # 如果给pair里边好的句子打分高（c_truncated_reward），坏的句子（r_truncated_reward）打分低，loss就会小：
            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()

            # print("T c_inds--5:", infoTensor(c_inds))
            # #only ph2, 很多次数，大小经常变化 T c_inds--5: _Size([72, 1])_int64_cuda:0_

            # print("T c_truncated_reward--5:", infoTensor(c_truncated_reward))
            # #only ph2, 很多次数，大小不定 T c_truncated_reward--5: _Size([48])_float16_cuda:0_

            # print("T check_divergence--5:", infoTensor(check_divergence))
            # #only ph2, 很多次数，大小不定 T check_divergence--5: _Size([0, 1])_int64_cuda:0_

            # print("T r_truncated_reward--5:", infoTensor(r_truncated_reward))
            # #only ph2, 很多次数，大小不定 T r_truncated_reward--5: _Size([111])_float16_cuda:0_

        # print("len of rejected_mean_scores", len(rejected_mean_scores))
        # #len of rejected_mean_scores 8
        # 将模型在整个batch中每个样本上计算得到的损失值进行平均，得到一个标量值作为这个batch的损失
        loss = loss / bs

        # 将两个分数列表堆叠为tensor
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)

        # print("chosen_mean_scores", chosen_mean_scores)
        # print("rejected_mean_scores", rejected_mean_scores)

        # print("T chosen_mean_scores--5:", infoTensor(chosen_mean_scores))
        # print("T rejected_mean_scores--5:", infoTensor(rejected_mean_scores))
        # only ph2
        # T chosen_mean_scores--5: _Size([8])_float16_cuda:1_
        # T rejected_mean_scores--5: _Size([8])_float16_cuda:1_

        # 返回包含损失值以及被选定和被拒绝序列的平均分数的字典
        # 返回损失和参考分值
        return {
            "loss": loss,
            "chosen_mean_scores": chosen_mean_scores,
            "rejected_mean_scores": rejected_mean_scores,
        }


    '''
    在训练强化学习的过程中，会用到reward model（critic model，
    再次提醒，critic model和reward model是同一个模型的两个副本）的推理过程，
    通过调用forward_value实现，具体代码如下，返回的值中有两种值，
    values表示每个位置i，从第i个位置到最后一个位置的奖励累加值，供强化学习过程中critic model使用；
    “chosen_end_scores”指的是对每个prompt+answer的打分，供reward model使用。
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

        #debuginfo(prj='ds-chat', info=self.__class__.__name__) #后面有细分

        # 模型的隐藏状态，它是一个列表，包含了每一层的输出，
        # 每一层的输出都是一个形状为[batch_size, sequence_length, hidden_size]的tensor。
        # 经过主干网络正向传播得到输出
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache)

        print("transformer_outputs is-2",  transformer_outputs)

        # hidden_states.shape: (bs, max_seq_len, hidden_size)
		# 取出第一层的隐藏状态
        hidden_states = transformer_outputs[0]

        # 将隐状态特征传入线性层v_head输出得到分值"""
        # values.shape: (bs, max_seq_len)
		# 这个值可以被视为模型对每个输入token的奖励
        values = self.v_head(hidden_states).squeeze(-1)

        ## value 为 bs * seq 的sequence 分数
        if return_value_only:
            debuginfo(prj='ds-chat', info=self.__class__.__name__)

            """
            如果传参中预设了“return_value_only=True”，
            那么将直接返回 values: (bs, max_seq_len)
            """
            return values
        else:
            debuginfo(prj='ds-chat', info=self.__class__.__name__)

            # 否则还将进一步取得reward_score
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            # 序列的开始部分通常包含一些padding，如果问题部分长度不超过1，则无法准确获取答案部分的值。
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"

            # batch size
            bs = values.size(0)
            # 序列长度
            seq_len = input_ids.shape[1]

            # 保存每个输入序列答案部分的最后一个值
            # we use this name for consistency with the original forward function
            chosen_end_scores = []
            for i in range(bs):
                # 获取当前序列的输入ID和模型计算出的值
                input_id = input_ids[i]

                # value.shape: (max_seq_len,)
                value = values[i]

                # 查找当前序列答案部分（即问题部分之后的部分，由input_id[prompt_length:]得到）中的padding标记。
                # 这一步的结果是一个包含所有padding标记位置的列表
                # c_ind即为prompt之后的序列片段中，第一个pad_token的index
                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()

                # 计算答案部分的结束位置，如果答案部分中存在padding标记（即len(c_inds) > 0），
                # 结束位置就是第一个padding标记的位置；否则，结束位置就是整个序列的长度。
                # 这里通过c_inds[0].item() + prompt_length来获取绝对位置，
                # 因为c_inds[0].item()只是相对于答案部分的开始位置的相对位置。
                # here we only use the answer part of the sequence so
                # we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len

                #取c_ind的前一个index（实际上就是answer的最终位置）作为reward_score
				# 将答案部分的最后一个值（即结束位置前一个位置的值，由value[c_ind - 1]获取）加入到chosen_end_scores列表中。
                chosen_end_scores.append(value[c_ind - 1])
				
            # 得到一个包含每个序列答案部分最后一个值的列表，这个列表可以用来表示每个序列的答案的"值"。
            # 返回values和reward_score"""
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }

        # 上面的Reward函数可以认为是一个分类器，需要基于Causal
        # LM（例如OPT）作为Backbone，获得完整的Reward模型： def create_critic_model(model_name_or_path
        # 上述reward model 用来训练奖励模型


'''
hidden_states is: tensor([[[-0.8462, -5.7422,  0.7568,  ..., -2.2910, -1.5664,  0.4041],
 ....
 [-0.8838, -3.8242,  0.0475,  ..., -0.6792,  0.3499,  3.2148]]], device='cuda:1', dtype=torch.float16)
rewards is: tensor([[-0.4866, -0.3369, -0.3350,  ..., -0.2542, -0.2542, -0.2542],
        ...,
        [-0.4866, -0.3369, -0.3350,  ..., -0.0409,  0.3579,  0.1017]], device='cuda:1', dtype=torch.float16)
'''

'''
chosen_id is: tensor([    2, 50118, 50118, 33837,    35,   653,   109,    47,  1669,   206,
        ...
           33,    10,   182,  2166,  2465,   516,     8,  5929],
       device='cuda:0')
rejected_id is: tensor([    2, 50118, 50118, 33837,    35,   653,   109,    47,  1669,   206,
        ...
           33,    10,   182,  2166,  2465,   516,     8,  5929],
       device='cuda:0')
chosen_reward is: tensor([-0.4866, -0.3369, -0.3350, -0.3252, -0.3557,  0.3831,  0.0515,  0.1240,
        ...
        -0.0613,  0.6763, -0.2629,  0.5146,  0.2266, -0.4478, -0.4583, -0.7490],
       device='cuda:1', dtype=torch.float16)
rejected_reward is: tensor([-0.4866, -0.3369, -0.3350, -0.3252, -0.3557,  0.1185,  0.6387,  0.4653,
        ...
        -0.7280,  0.4292,  0.0875,  0.0076,  0.2520,  0.3594, -0.3235,  0.1514],
       device='cuda:0', dtype=torch.float16)
'''

'''
c_ind is: 87
c_inds is: tensor([[ 50],
        [ 51],
...
        [ 99],
        [100]], device='cuda:0')
c_truncated_reward is: tensor([[  0],
        [ 83],
...
        [126],
        [127]], device='cuda:1')
check_divergence is: tensor([-0.7969, -0.9590, -0.7002, -0.7397, -1.0400, -1.1787, -1.1152, -1.8379,
        -2.3906, -2.6797, -2.9160, -2.8047, -3.2949, -3.1367, -2.9453, -2.7754,
...
        -1.4961, -1.4961, -1.4961], device='cuda:0', dtype=torch.float16,
       grad_fn=<SliceBackward0>)
r_truncated_reward is: tensor([-1.2334, -1.8516, -2.3652, -2.6016, -2.6055, -2.8008, -2.8809, -2.7012,
        -3.0293, -2.9863, -3.3477, -3.1641, -2.8906, -3.1367, -3.3398, -3.4160,
...
        -2.3105, -2.2109, -2.3320], device='cuda:0', dtype=torch.float16,
       grad_fn=<SliceBackward0>)
'''

'''
chosen_mean_scores tensor([-0.2749,  0.4365,  0.0247,  0.9805,  1.0439,  1.0908, -0.4592,  0.3604],
        device='cuda:0', dtype=torch.float16, grad_fn=<StackBackward0>)
rejected_mean_scores tensor([-0.3792,  1.2471,  0.5435, -1.0049,  0.5679, -0.0025,  0.4866,  1.1855],
       device='cuda:1', dtype=torch.float16, grad_fn=<StackBackward0>)
'''