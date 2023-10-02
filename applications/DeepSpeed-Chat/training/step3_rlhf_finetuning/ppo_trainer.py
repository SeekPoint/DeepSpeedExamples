# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total

'''
3.3.3.4 策略模型logits的进一步处理
策略模型（actor、ref/SFT）所输出logits的shape为(bs, max_seq_len, vocab_size)，
然而计算KL散度惩罚、重要性权重时并不需要对所有vocab的logits进行计算，
仅需要对groundtruth项（seq各个token对应的项）的logits进行计算即可。

batch_size = 1
max_seq_len = 4
vocab_size  = 3

logits = [
		  [[1.23, 2.11, -0.56], 
		   [-1.52, -1.11, 1.66], 
		   [0.32, 0.13, 1.55], 
		   [-0.55, -0.23, -1.62]]
		 ]

seq = [
	   [2, 2, 0, 1]
	  ]

对于CausalLM来说，
logits第t个时间步的置信值是为了预测第t+1步的seq token，
因此logits[, :-1, :]与seq[:, 1:]才是“预测与标签”的关系：
logits[, :-1, :] = [
		  			[[1.23, 2.11, -0.56], 
		  			 [-1.52, -1.11, 1.66], 
		   			 [0.32, 0.13, 1.55]]
		 		   ]
seq[:, 1:] = [
			  [2, 0, 1]
			 ]
只需要从预测中根据对应标签取出logits即可，
以上述例子为例，最终取出的结果probs为
probs = [
			 [-0.56, -1.52, 0.13]
			]
			
因此DeepSpeed-Chat定义了函数gather_log_probs()来对输出的logits进行后处理，以获取对数化后的结果log_probs。
'''
def gather_log_probs(logits, labels):
    """
    相当于输入logits和labels，对logits进行log_softmax后取出对应label位置耳朵logit值
    :param logits: (bs, seq_len, vocab_size)
    :param labels: (bs, seq_len)
    :return: log_probs_labels.squeeze(-1): (bs, seq_len)
    """
    ## log_probs.shape: (bs, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)

    """
    此处gather()可以根据labels（index）来从log_probs中获取对应index的值
    总的来说就是取出logits中对应labels数值位置的值
    log_probs_labels.shape: (bs, seq_len, 1)
    """
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

'''
强化学习微调
强化学习微调阶段，会用到4个模型，actor model， ref_model，reward model和critic model（好费显存啊！！！）。
其中actor model和ref_model是RLHF第一个阶段有监督微调模型的两个副本，reward model和critic model是本文第一部分训练出来的模型的两个副本。
整体流程见这篇文档，整体流程图如下所示（没画出critic model）：

图！

首先说明actor model的训练模式和推理模式的区别（ 后边会用到）。
训练模式是用teacher force的方式（不明白的同学知乎搜一下），将整句话输入到模型中，并通过mask机制在保证不泄漏未来的单词情况下预测下一个单词。
推理模式是真正的自回归，预测出下一个单词之后，当作下一步输入再预测下下个单词，原理如下图所示：

图！

'''
class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.end_of_conversation_token_id = self.tokenizer(
            args.end_of_conversation_token)['input_ids'][-1]
        self.z3_enabled = args.actor_zero_stage == 3

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
'''
3.3.3.2 seq的生成
对于本次batch的prompt，将输入至当前actor（对于即将根据经验数据迭代得到的actor来说，
此时的“当前actor”可以认为是“旧策略网络”）来生成answer（如下图所示），
然后将prompt与answer进行拼接得到seq。
在这里插入图片描述  011.png
'''
    def _generate_sequence(self, prompts, mask, step):
        """
        生成seq

        获取prompt拼接上answer后的最大长度，
        实际上相当于max_seq_len，
        用于对生成长度做限制
        """
        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # 首先用actor model在推理模式下根据prompt生成一个answer（prompt对应强化学习里边的state，answer对应一些列的action），代码如下：
        # 保证不触发反向传播
        with torch.no_grad():
            #调用actor，输入input_ids和attention_mask进行生成
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled)

        # """下方操作是为了过滤掉只有极短answer（有效长度小于1）的seq"""
        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]

        #prompt长度：实际上就是max_prompt_len
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length

        #取出answer部分，此时还含有pad token
        ans = seq[:, prompt_length:]

        #统计answer的有效长度（去掉pad token后的长度）
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        if self.args.print_answers:
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        #排除较短（此处设置为有效长度小于1）的answer，余下的answer将被存入out_seq作为最终返回
        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq

    '''
    3.3.3.1 经验数据获取过程
    010.png
    
    再次借用ColossalChat的示意图来进行说明，经验数据的获取过程如下：
    
        1备有prompt数据（prompt_input_ids，prompt_attention_mask）；
        2使用当前actor对prompt进行answer生成，得到完整对话序列seq（图示的sequence）；
        3将seq输入至当前actor，输出得到当前（旧）策略logits（图示的action_logits），取对数logprobs；
        4将seq输入至ref/SFT，输出得到baseline策略ref_logits（图示的sft_logits），取对数ref_logprobs；
        5将seq输入至reward/RM，输出得到环境奖励reward_score（图示的r(x,y)）；
        6将seq输入至当前critic，输出得到当前（旧）价值估计values（图示的value）；
        7至此，用于进行PPO训练的各个基本经验数据已经获取齐全，
        至于图示的adv、reward（此reward非彼reward，图示的reward指InstructGPT所提及的“KL Reward”：
        为了防止对phase2学习到的reward过度自信，引入了SFT与logits的KL散度作为惩罚的Reward）等数据，
        在DeepSpeed-Chat中，于具体训练过程才开始计算。
    
    相关代码实现可见下方代码块。
    '''
    def generate_experience(self, prompts, mask, step):
        '''
        生成经验
        :param prompts: prompt input ids，(bs, max_prompt_len)
        :param mask: prompt attention mask, (bs, max_prompt_len)
        :return:
        '''

        #将actor、reference、critic、reward转换为eval模式
        # 给定prompt，生成response text
        # 开启eval模式
        self.eval()

        '''
        seq.shape: (seq_bs, max_seq_len)
        seq_bs指：排除较短answer后的batch_size。
        所谓“较短answer”在默认设定中是“序列长度小于1的answer”，
        短answer的seq都被滤掉了，
        所以可能batch_size会比之前小，
        但这个可能性极低，DS-C认为只有在使用未经phase1训练的模型来生成才会出现该情况。
        
        _generate_sequence()更具体的细节可见后续详解。
        '''
        # 调用model.generate()生成序列，由actor模型生成。
        # 输入instruct prompt，由Actor生成seq，上图中红色步骤（1），seq由instruct和response组成
        seq = self._generate_sequence(prompts, mask, step)

        #将actor、critic转换为train模式，因为后续两者仍需要进行训练
        # 恢复训练模型
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()
        '''
        actor model是我们想通过强化学习微调的大模型，但是强化学习过程很容易把模型训练“坏”，
        因此需要另外一个不会参数更新的 ref_model来当作标的，别让actor mode跑偏太远。
        我们在训练模式下，将prompt+answer分别输入到actor mode和ref model，用KL散度来衡量 ref model和actor mode输出的差别。
        同时将KL散度（衡量数据分布差距大小）纳入损失函数（KL散度本质是纳入到奖励值里边的，奖励值被纳入到了损失函数），进而来约束 ref_model和actor mode的输出分布别差距太大。
        具体代码如下：
        '''
        with torch.no_grad():
            # 得到两个模型的输出
            '''
            经验采集：这部分其实就是在获取计算phase3损失函数所需的内容
            1. actor:（旧）策略-output.logits
            2. reference:SFT策略-output_ref.logits
            3. reward:奖励-reward_score，InsructGPT中的r_\theta
            4. critic:（旧）价值估计-values
            '''
            # 将生成的序列喂入actor模型中，得到输出的概率分布
            # 将seq喂入actor中得到action_logits，上图中棕色步骤（2）
            output = self.actor_model(seq, attention_mask=attention_mask)

            # 将seq喂入SFT中得到sft_logits，上图中黑色步骤（5）
            output_ref = self.ref_model(seq, attention_mask=attention_mask)

            #然后利用reward model和ciric model对输出的prompt+answer进行打分
            # （PPO训练时使用的奖励值并不单单是reward model的输出还要考虑kl散度，后文介绍）：
            # 奖励模型返回的是个字典，key为chosen_end_scores位置存储数据维度为(B,)，表示对于prompt+answer的打分
            '''
            价值函数的forward_value()更具体的细节可见后续详解。
            reward_score取的是answer最后一个token的value
            reward_score.shape: (seq_bs,)
            '''
            # 将生成的序列喂入critic和reward模型中，获得奖励和状态价值
            # 将seq喂入reward模型中打分，得到r(x,  y)，上图绿色步骤（4）
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )

            # critic model返回的数据维度为(B,L)，L维度上第i个位置代表从i位置到最后的累积奖励
            # 舍去最后一个位置是因为句子“终止符”无意义

            # critic_model.forward_value(return_value_only=True)将返回shape为(seq_bs, max_seq_len)的序列各token的value
            # 将seq喂入critic，获得critic的value，上图蓝色步骤（3）
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

        # (seq_bs, max_seq_len, vocab_size)
        logits = output.logits

        # (seq_bs, max_seq_len, vocab_size)
        logits_ref = output_ref.logits

        # 返回的dict是“进行PPO所需要使用的一组数据”
        # prompts.shape: (bs, max_prompt_len)
        # logits[:, :-1, :].shape: (seq_bs, max_seq_len - 1)
        # seq[:, 1:].shape: (seq_bs, max_seq_len - 1)
        # gather_log_probs()相当于输入logits和labels，对logits进行log_softmax后取出对应label位置的logit值
        # 因此logprobs.shape: (seq_bs, max_seq_len - 1)，ref_logprobs.shape: (seq_bs, max_seq_len - 1)
        # values.shape: (seq_bs, max_seq_len - 1)
        # rewards.shape: (seq_bs,)，reward_score在InstructGPT中就是r_\theta
        # input_ids.shape: (seq_bs, max_seq_len)
        # attention_mask.shape: (seq_bs, max_seq_len)
        """gather_log_probs()更具体的细节可见后续详解。"""
        # 获得生成的文本seq、以及对应的概率、状态价值和奖励等信息
        # 获得经验数据
        return {
            'prompts': prompts,
            # 分别得到两个模型在真实单词上的预测概率
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,
                                                                        1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):
        # 计算kl散度，log_probs里边存的数字经过log变化了，因此减法就对应除法
        """
        计算实际rewards，涉及（旧）策略与SFT的KL散度惩罚、RM的reward
        计算经验采样时actor与SFT的KL散度惩罚
        """
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)

        # PPO训练时候的奖励值综合考虑KL散度和reward模型的输出，只考虑answer部分的KL散度，将reward
        # model的输出加到KL散度L维度的最后一个位置上，得到最终的奖励值，代码如下：
        rewards = kl_divergence_estimate

        # 只考虑answer部分的奖励，不考虑prompt
        """
        找到answer的起始start：即prompt的最后1个token位置
        比如prompts长度为256，answer的起始则为256-1=255
        """
        start = prompts.shape[1] - 1

        # 不考虑padding部分
        '''
		ends为batch中各个数据的最后1个有效token的index，
		每个数据的最末有效token位置很大可能是不一样的，
		因此ends是个数组
		'''
        ends = start + action_mask[:, start:].sum(1) + 1

        #将RM得到的奖励值限定在一定范围，默认为(-5,5)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]

        # 在L维度上，每个位置都有KL散度，但是只在最后一个位置加上奖励值
        '''
        因为batch中每个数据的最末有效token位置很可能不一样，
		所以无法通过矩阵来并行，需要使用for循环逐个数据处理
		'''
        for j in range(batch_size):
            """
            KL_reward = KL + reward
            加和只在最末有效token上进行
            """
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        """返回KL rewards"""
        return rewards

    '''
    3.3.5.2 PPO训练
    012.png
    1次PPO训练由train_rlhf()方法进行管理，其内部主要实现了：
    013.png
    具体代码可见下方，为保证阅读的流畅性，我对其中的部分代码进行了调整，
    使得相应的函数代码衔接在其调用后方，便于具体对照其传参，从而辨析传入的新旧策略、新旧价值估计等：
    '''
    def train_rlhf(self, inputs):
        # 使用1个ppo_batch的经验数据，执行1次rlhf训练迭代

        # train the rlhf mode here
        ### process the old outputs
        # prompt input ids
        # 当前RLHF轮次最初采样的经验池中采样一批数据
        # instruct prompt
        prompts = inputs['prompts']  # 输入的prompt（例如in-context exemplar + query）

        # （旧）策略
        # actor模型生成response对应的action_logist
        log_probs = inputs['logprobs'] # 根据prompt，actor模型生成的文本的概率

        # SFT策略
        # SFT模型生成response对应的sft_logits
        ref_log_probs = inputs['ref_logprobs']  # 根据prompt，reference生成模型的文本的概率

        # RM奖励
        # reward模型预测的奖励r(x, y)
        reward_score = inputs['rewards']  # 根据prompt生成的seq，reward模型得到的奖励

        # （旧）价值估计
        # critic模型预测的奖励
        values = inputs['value']  # 根据prompt生成的seq，critic模型得到的状态价值函数值
        attention_mask = inputs['attention_mask']  # actor生成的文本的attention mask

        # seq input ids
        seq = inputs['input_ids'] # 根据prompt，actor生成的文本

        """
        获取prompts的最后1个位置作为start
        比如prompt_len为256，start则为 256-1=255
        这个start主要是用于取出经验数据中的“非prompt”部分（也即“answer+padding”部分）
        """
        start = prompts.size()[-1] - 1   # 记prompt文本最后一个位置

        """
        action_mask相当于取 attention_mask除了第0个序列位置外的部分，
        需要注意的是：
        1. 多数情况下，包括此处在内的transformers风格代码中，
        attention_mask指的实际上是“padding_mask”而非“sequence_mask”；
        2. 之所以要进行[:, 1:]切片，是为了去除第0个位置从而与seq对齐，
        因此 action_mask.shape: (bs, max_seq_len - 1)
        3. 后续将被用于过滤掉pad token位置的信息
        4. 但实际上在后续的使用中，
        基本都会结合上方定义的start，从action_mask中再切片出“非prompt”部分，
        例如 action_mask[start:]，实际上就相当于取“非prompt”部分，
        action_mask[start:].shape: (bs, max_answer_len)
        """
        action_mask = attention_mask[:, 1:]

        # 经验数据中的价值估计为“旧”价值估计
        ### 根据经验数据，接下来计算相应的reward和advantage
        old_values = values
        with torch.no_grad():
            ###计算KL惩罚修正的奖励################################################

            """
            通过KL散度惩罚，以及r_\theta（来自reward model）计算得到修正的奖励，
            注意此处的入参：
            1. log_probs为经验数据中的旧策略
            2. ref_log_probs为经验数据中的SFT策略
            3. reward_score为经验数据中的RM赋分
            """

            # 获得prompt文本本身的奖励
            # 由于prompt本身已存在文本，相当于整个决策序列中中已有的状态动作序列，
            # 因此我们需要计算一下prompt文本对应的奖励

            # 根据SFT的sft_logits和Actor的action_logist，计算KL散度；
            # 并根据KL散度与reward模型预测的奖励r(x, y)，获得最终奖励
            # 上图中红色步骤（1）
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0

            '''
            计算优势与回报################################################
            计算优势advantages和回报returns
            注意此处的入参：
            4. old_value为经验数据中的（旧）价值估计
            5. old_rewards为刚才计算得到的KL_reward
            '''
            # 获得advantage值（v + r - v'）
            # 由critic或的的value与前面根据KL散度和r(x, y)得到的reward，从而计算得到advantage
            # 上图蓝色步骤（2）
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        # ###计算actor损失并更新
        # 下面则是获得生成部分seq的奖励等信息
        ### 根据经验数据以及得到的advatage，下面开始获得一系列的loss
        batch = {'input_ids': seq, "attention_mask": attention_mask}

        #将seq经验数据输入至actor，进行自回归预测
        # 获得seq的的概率

        # 将这一批经验数据的seq（instruct prompt+response）再一次喂入actor得到logits
        # 因为现在是在更新actor和critic，而经验数据所采用的actor和critic早已经是之前的了，所以
        # 现在正在更新的actor和critic与当时进行经验采样时的actor、critic的参数已经有差异了；
        # 所以需要重新获得当前最新的actor输出的logits
        # 上图中棕色步骤（3）
        actor_prob = self.actor_model(**batch, use_cache=False).logits

        #取出probs，此处为新策略
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])

        """
        计算actor损失
        注意此处的入参：
            1. actor_log_probs为方才刚输出的新策略
            2. log_probs为经验数据中的（旧）策略
            3. advantages为之前计算出的优势
        """
        # 根据seq的概率logits，advantage作为权重，优化actor模型参数

        # 根据新的actor logits以及经验数据中的logits，以及advantage，计算actor loss
        # 上图中绿色步骤（4）
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])

        #actor反向传播、更新参数
        # 更新actor参数
        # 更新actor模型参数
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        #计算critic损失并更新################################################
        #将seq经验数据输入至critic，预测得到新价值估计
        # 获得seq的critic得分

        # 经验数据中的seq（instruct prompt+response）再一次喂入critic得到value
        # 同理，由于当前的critic和当初进行经验数据采样时的critic相差很远；所以需要重新获得value
        # 上图中黑色步骤（5）
        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]

        """
        计算critic损失
        注意此处的入参：
           1. values为方才刚输出的新价值估计
           2. old_values为经验数据中的（旧）价值估计
           3. returns为之前计算出的回报
        """
        # 计算Critic loss
        # 根据最新的critic的value，经验数据的old_value，以及advatage，计算得到critic loss
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])

        #critic反向传播、更新参数
        # 更新Critic模型参数
        # 更新critic参数
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            rank = torch.distributed.get_rank()
            if actor_overflow and not critic_overflow:
                self.critic_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: actor overflow, skipping both actor and critic steps",
                    rank)
            elif not actor_overflow and critic_overflow:
                self.actor_model.optimizer.skip_step = True
                print_rank_0(
                    "OVERFLOW: critic overflow, skipping both actor and critic steps",
                    rank)
            elif actor_overflow and critic_overflow:
                print_rank_0(
                    "OVERFLOW: actor and critic overflow, skipping both actor and critic steps",
                    rank)
            self.actor_model.step()

        self.critic_model.step()

        # 本次ppo_step将返回actor_loss和critic_loss供指标统计
        return actor_loss, critic_loss

    '''
    以上过程，我们已经拿到了PPO训练所需要的advantage以及actor model的输出，我先现在可以对actor model进行训练啦。
    具体代码如下。logprobs和old_logprobs这两个参数分别是“老actor（n个epoch才会更新一次）”和新actor（每个batch都会更新它）”在正确单词上出处的概率，
    这块时PPO import sampling相关的知识，就不在这重复介绍了，不明白的同学补习一下哈。借用一下李宏毅老师的PPO公式：
    图！


    '''
    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        #"""计算actor的损失"""

        ## policy gradient loss
        # logprobs, old_logprobs都是经过log变化的单词概率，这里带着log做减法就相当于在做概率除法

        # 重要性采样权重计算：ratio = exp(log(new)-log(old))
        log_ratio = (logprobs - old_logprobs) * mask

        # 指数操作去掉log
        ratio = torch.exp(log_ratio)

        #计算策略梯度损失的2个情况：加权优势 与 裁剪加权优势
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)

        # 从2个情况中选择损失较大者作为真正的损失，
        # 并且基于ppo_batch内所有数据的所有有效时间步计算平均损失值
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    #同样的，我们也要对critic model进行训练，更新，loss就是mse loss。
    def critic_loss_fn(self, values, old_values, returns, mask):
        # 计算价值损失

        ## value loss
        ## 用“老critic model”的输出约束“新critic model”不要步子太大，裁剪一下
        '''
        至此，我们的RLHF训练流程就结束了。第二部分开头我们说过，共涉及actor model， ref_model，reward model和critic model这四个模型，
        其实更新参数的模型只有actor model和critic model。

        裁剪当前新values，使得其不至于太偏离经验采样阶段的旧values
        '''
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )

        #计算当前values与回报的L2 Loss
        vf_loss1 = (values - returns)**2

        #计算裁剪后的当前values与回报的L2 Loss
        vf_loss2 = (values_clipped - returns)**2

        """
        选择损失较大者作为真正的损失，
        并且基于ppo_batch内所有数据的所有有效时间步计算平均损失值，
        此外critic损失项的系数为0.5。
        """
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    '''
    接下来的内容就是PPO的训练过程的比较核心的内容了，目标是计算PPO更新公示里边的advantage，
    具体公式如下，V就是critic model的输出。如果原理不懂建议先到这个链接
    https://link.zhihu.com/?target=https%3A//huggingface.co/blog/deep-rl-a2c
    看看。我直接在代码中给注释了。
    图！

    图片出处：https://huggingface.co/blog/deep-rl-a2c
    '''
    def get_advantages_and_returns(self, values, rewards, start):
        # values（B，L） critic model输出
        # rewards（B，）reward model输出
        # start answer开始的位置
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134

        # 计算优势与回报
        # 实现基本与上述公式相同
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]

        # 计算每个时刻（序列位置）的critic model预测误差
        # 反向遍历计算各个时间步的优势advantage
        for t in reversed(range(start, length)):
            # 获取下个时间步的价值估计V_{old}(s_{t+1})
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0

            # critic model预测的是t到到最后一个时刻的奖励和，所以变化量delta可以用如下公式表示
            """计算单步TD-error"""
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]

            """累计优势"""
            # self.gamma=1，self.lam=0.95是衰减因子，表示之前计算的delta对现在影响越来越小
            lastgaelam = delta + self.gamma * self.lam * lastgaelam

            # """存储各个时间步的优势"""
            advantages_reversed.append(lastgaelam)

        #对逆序的优势列表进行正序处理，得到正常时间步排列的优势
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        # 后续用来更新critic model用
        """
        return_t = adv_t + v(s_t)
        由优势计算得到回报
        """
        returns = advantages + values[:, start:]

        # 返回优势与回报
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_critic_model_norm', critic_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_reward_model_norm', reward_model_norm,
                        self.args.local_rank)

'''
3.3.5.3 无监督训练
实际上就是常规的自回归语言建模任务。InstructGPT中提及，进行phase3的RLHF训练时，
为使得模型在学习人类偏好的过程中仍能保有预训练模型解决任务的性能，
引入了传统的自回归语言建模进行联合训练。
014.png
'''
class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        """
        1个ppo_batch的无监督训练
        :param inputs: dict：input_ids, attention_mask, labels
        :param unsup_coef: 无监督损失系数

        确保actor处于训练模式，否则将返回报错
        """

        # Train the unsupervised model here
        self._validate_training_mode()

        #actor进行常规的CausalLM训练
        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss

        # 反向传播、更新参数
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
