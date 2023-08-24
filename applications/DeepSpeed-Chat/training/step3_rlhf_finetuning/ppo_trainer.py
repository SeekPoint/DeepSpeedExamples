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


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
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

    def _generate_sequence(self, prompts, mask, step):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # 首先用actor model在推理模式下根据prompt生成一个answer（prompt对应强化学习里边的state，answer对应一些列的action），代码如下：
        # 保证不触发反向传播
        with torch.no_grad():
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled)

        # Filter out seq with no answers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        if self.args.print_answers:
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[
                    i] <= 1:  # if the answer is shorter than 1 token, drop it
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim

        return out_seq

    def generate_experience(self, prompts, mask, step):
        self.eval()
        seq = self._generate_sequence(prompts, mask, step)
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
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            #然后利用reward model和ciric model对输出的prompt+answer进行打分（PPO训练时使用的奖励值并不单单是reward model的输出还要考虑kl散度，后文介绍）：
            reward_score = self.reward_model.forward_value(
                seq, attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach(
                )
            # critic model返回的数据维度为(B,L)，L维度上第i个位置代表从i位置到最后的累积奖励
            # 舍去最后一个位置是因为句子“终止符”无意义
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits

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
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)

        # PPO训练时候的奖励值综合考虑KL散度和reward模型的输出，只考虑answer部分的KL散度，将reward
        # model的输出加到KL散度L维度的最后一个位置上，得到最终的奖励值，代码如下：
        rewards = kl_divergence_estimate

        # 只考虑answer部分的奖励，不考虑prompt
        start = prompts.shape[1] - 1

        # 不考虑padding部分
        ends = start + action_mask[:, start:].sum(1) + 1
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]

        # 在L维度上，每个位置都有KL散度，但是只在最后一个位置加上奖励值
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            ends = start + action_mask[:, start:].sum(1) + 1
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                old_rewards[i, ends[i]:] = 0
                old_values[i, ends[i]:] = 0
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)

        if not self.args.align_overflow:
            self.actor_model.step()

        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                                       start:],
                                          returns, action_mask[:, start:])
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

        return actor_loss, critic_loss

    '''
    以上过程，我们已经拿到了PPO训练所需要的advantage以及actor model的输出，我先现在可以对actor model进行训练啦。
    具体代码如下。logprobs和old_logprobs这两个参数分别是“老actor（n个epoch才会更新一次）”和新actor（每个batch都会更新它）”在正确单词上出处的概率，
    这块时PPO import sampling相关的知识，就不在这重复介绍了，不明白的同学补习一下哈。借用一下李宏毅老师的PPO公式：
    图！


    '''
    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        # logprobs, old_logprobs都是经过log变化的单词概率，这里带着log做减法就相当于在做概率除法
        log_ratio = (logprobs - old_logprobs) * mask

        # 指数操作去掉log
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    #同样的，我们也要对critic model进行训练，更新，loss就是mse loss。
    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        ## 用“老critic model”的输出约束“新critic model”不要步子太大，裁剪一下
        '''
        至此，我们的RLHF训练流程就结束了。第二部分开头我们说过，共涉及actor model， ref_model，reward model和critic model这四个模型，
        其实更新参数的模型只有actor model和critic model。

        '''
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
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
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        # 计算每个时刻（序列位置）的critic model预测误差
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            # critic model预测的是t到到最后一个时刻的奖励和，所以变化量delta可以用如下公式表示
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            # self.gamma=1，self.lam=0.95是衰减因子，表示之前计算的delta对现在影响越来越小
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # 后续用来更新critic model用
        returns = advantages + values[:, start:]
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


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
