# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import sys
import os
import deepspeed
# ZeroParamStatus : 用于在DeepSpeed的Zero Redundancy Optimizer（零冗余优化器）中跟踪参数状态的工具
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

# 将父目录的路径添加到了系统路径中
# os.path.dirname(__file__) : 获取当前文件的目录
# os.path.pardir 是一个特殊的常量，表示父目录
# os.path.abspath 返回参数的绝对路径
sys.path.append(
    # 获取了当前文件的父目录的绝对路径
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from utils.utils import print_rank_0
from pydebug import gd, infoTensor

def print_all_ranks(tag, value, rank):
    '''在多进程（即分布式）训练的环境中，将各个进程（rank）的某个值打印出来。'''

    # 分布式训练的进程总数
    world_size = torch.distributed.get_world_size()

    # 初始化一个全零的张量
    all_tensor = torch.zeros(world_size, dtype=torch.float32).cuda()

    # 将当前进程的值存入all_tensor的对应位置
    all_tensor[rank] = value

    # 分布式通信操作，将所有进程的all_tensor进行汇总，得到一个新的all_tensor，
    # 其元素是所有进程的all_tensor的对应元素之和。
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)

    # 打印出这个all_tensor，以及一个标签tag
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    '''计算并返回整个模型的参数范数（norm）
       定义 : 范数是一个可以度量向量空间中元素大小的函数，这里用来度量模型参数的大小。
       原因 :
        ① 监控模型训练：参数范数可以用于监控模型训练过程。
        ② 防止过拟合：如果参数的范数变得非常大，模型可能出现过拟合现象。
        ③ 控制模型复杂度：参数范数也可以视为一种模型复杂度的度量。
        ④ 配合优化器使用：某些优化器（例如 Adam）会跟踪模型参数的范数来调整学习率。
    '''
    with torch.no_grad():
        # 累计所有参数的范数
        total = 0.0

        # 遍历模型的所有参数
        for param in model.parameters():
            # 检查每个参数param是否具有ds_id属性并且其ds_status属性是否为NOT_AVAILABLE
            # 原因 : 这主要是针对使用了DeepSpeed的ZeRO的情况，这种优化器可以分布式地存储和更新模型的参数，
            #       而不需要每个进程都保持一份完整的参数。因此，一些参数可能不在当前进程中，
            #       这就需要通过ds_id和ds_status这些属性进行检查。
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE

            # 如果参数在当前进程中不可用（即ds_status为NOT_AVAILABLE），则需要收集（gather）它
            print("####### 1- with deepspeed.zero.GatheredParameters #####################")
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                print("####### 2- with deepspeed.zero.GatheredParameters #####################")
                # 计算参数的范数并累加到total中
                total += float(param.float().norm())
            print("####### 3- with deepspeed.zero.GatheredParameters #####################")

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
    '''从模型输出的logits中收集每个标签对应的对数概率'''
    """
    相当于输入logits和labels，对logits进行log_softmax后取出对应label位置耳朵logit值
    :param logits: (bs, seq_len, vocab_size)
    :param labels: (bs, seq_len)
    :return: log_probs_labels.squeeze(-1): (bs, seq_len)
    """
    # log_probs.shape: (bs, seq_len, vocab_size)
	# log_softmax操作可以将logits转化为对数概率
    log_probs = F.log_softmax(logits, dim=-1)

    """
    此处gather()可以根据labels（index）来从log_probs中获取对应index的值
    总的来说就是取出logits中对应labels数值位置的值
    log_probs_labels.shape: (bs, seq_len, 1)
    """
    # 使用torch.gather方法从log_probs中收集每个标签对应的对数概率
    # dim=-1 表示沿着最后一个维度（类别的维度）进行收集
    # index=labels.unsqueeze(-1) 表示收集的索引是labels对应的位置
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))

    # 使用squeeze操作移除大小为1的维度，这是因为unsqueeze(-1)会给labels增加一个维度，
    # 所以需要用squeeze将其移除，使得log_probs_labels的维度和labels一致，这样就得到了每个标签对应的对数概率。
    return log_probs_labels.squeeze(-1)

'''
强化学习微调
强化学习微调阶段，会用到4个模型，actor model， ref_model，reward model和critic model（好费显存啊！！！）。
其中actor model和ref_model是RLHF第一个阶段有监督微调模型的两个副本，
reward model和critic model是本文第一部分训练出来的模型的两个副本。
整体流程见这篇文档，整体流程图如下所示（没画出critic model）：

图！

首先说明actor model的训练模式和推理模式的区别（ 后边会用到）。
训练模式是用teacher force的方式（不明白的同学知乎搜一下），将整句话输入到模型中，
并通过mask机制在保证不泄漏未来的单词情况下预测下一个单词。
推理模式是真正的自回归，预测出下一个单词之后，当作下一步输入再预测下下个单词，原理如下图所示：

图！

'''
class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        self.rlhf_engine = rlhf_engine # rlhf引擎实例，包含了所有训练相关模型和参数
        self.actor_model = self.rlhf_engine.actor # actor模型用来决定要执行的动作
        self.critic_model = self.rlhf_engine.critic # critic模型用来评估actor选择的动作的价值
        self.ref_model = self.rlhf_engine.ref # ref模型
        self.reward_model = self.rlhf_engine.reward # reward模型，用于计算奖励
        self.tokenizer = self.rlhf_engine.tokenizer # tokenizer，用于文本的编码和解码

        self.args = args  # 传入的参数

        # 执行生成答案时，最大的答案序列长度
        self.max_answer_seq_len = args.max_answer_seq_len
		
        # 定义了会话结束的标志 # 对话结束的 token 索引
        self.end_of_conversation_token_id = self.tokenizer(args.end_of_conversation_token)['input_ids'][-1] 
        self.z3_enabled = args.actor_zero_stage == 3

        # Those value can be changed
        # 一些PPO训练的超参数
        # self.kl_ctl : KL散度控制参数，用于限制策略更新的步长。
        # ① 知识点：在实际训练过程中，如果策略更新的步长过大，
        #   可能会导致模型快速收敛但局部最优，或者模型完全不稳定。
        #    通过限制KL散度，可以使策略迭代更加稳定。
        self.kl_ctl = 0.1

        # self.clip_reward_value : 奖励剪裁值，用于限制奖励的大小。
        # ② 知识点：在强化学习中，如果奖励过大或过小，可能会导致训练不稳定。
        #          通过剪裁奖励，可以使得奖励在一定的范围内，从而提高训练的稳定性。
        self.clip_reward_value = 5

        # self.cliprange：策略剪裁范围，用于PPO（Proximal Policy Optimization）算法中的策略剪裁。
        # ③ 知识点：PPO通过限制策略的改变量，以保证新策略不会偏离旧策略太远，提高训练的稳定性。
        self.cliprange = 0.2

        # self.cliprange_value：价值剪裁范围，也是用于PPO中，与self.cliprange类似，但是它是对价值函数的剪裁。
        self.cliprange_value = 0.2

        # self.gamma：未来奖励的衰减因子，用于计算未来奖励的累计值。
        # ④ 知识点：如果gamma较大，表示对未来的奖励给予更大的重视；如果gamma较小，表示更重视即时奖励。
        self.gamma = 1.0

        # self.lam：是GAE（Generalized Advantage Estimation）中的参数，用于计算优势函数。
        # ⑤ 知识点：lam越大，表示越重视未来的奖励；lam越小，表示越重视即时奖励。
        self.lam = 0.95

    '''
    3.3.3.2 seq的生成
    对于本次batch的prompt，将输入至当前actor（对于即将根据经验数据迭代得到的actor来说，
    此时的“当前actor”可以认为是“旧策略网络”）来生成answer（如下图所示），
    然后将prompt与answer进行拼接得到seq。
    在这里插入图片描述  011.png
    '''
    def _generate_sequence(self, prompts, mask, step):
        '''生成一段对话序列，使用的是模型的generate方法，根据给定的prompt和mask生成对应的答案。'''
        """
        生成seq

        获取prompt拼接上answer后的最大长度，
        实际上相当于max_seq_len，
        用于对生成长度做限制
        """
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        # 最大答案序列长度加上给定提问的长度
        max_min_length = self.max_answer_seq_len + prompts.shape[1]
        # print("max_min_length--2 is:", max_min_length)
        # max_min_length--2 is: 512

        ## 首先inference获取对应的模型输出

        # 首先用actor model在推理模式下根据prompt生成一个answer
        # （prompt对应强化学习里边的state，answer对应一些列的action），代码如下：
        # 保证不触发反向传播
        with torch.no_grad():
            # 调用actor，输入input_ids和attention_mask进行生成
			# 生成序列，每个元素都是对应单词的token ID
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_length=max_min_length,  # 生成的答案序列长度会和问题序列长度一致
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled)
            # print("seq--2 is:", seq)
            # print("T seq--2  :", infoTensor(seq)) #only ph3 x1

        # """下方操作是为了过滤掉只有极短answer（有效长度小于1）的seq"""
        # Filter out seq with no answers (or very short).
        # This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
		# 批处理的大小
        batch_size = seq.shape[0]
        # print("batch_size--2 is:", batch_size)

        #prompt长度：实际上就是max_prompt_len
		# 提问的长度
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        # print("prompt_length--2 is:", prompt_length)

        #取出answer部分，此时还含有pad token
		# 去掉了输入的提问部分，只保留了模型生成的答案部分。
        ans = seq[:, prompt_length:]
        # print("ans--2 is:", ans)

        #统计answer的有效长度（去掉pad token后的长度）
		# 每个答案的有效长度，即非填充部分的长度。
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        # print("valid_ans_len--2 is:", valid_ans_len)


        # print("T ans--2 :", infoTensor(ans))  #only ph3 x1
        # print("T valid_ans_len--2 :", infoTensor(valid_ans_len)) #only ph3 x1
        '''
        T seq--2  : _Size([4, 512])_int64_cuda:0_   
        T ans--2 : _Size([4, 256])_int64_cuda:0_
        T valid_ans_len--2 : _Size([4])_int64_cuda:0_
        '''
        '''
        seq--2 is: tensor([[    2,     2,     2,  ...,  1079,     9,    39],
                [    2,     2,     2,  ...,    47,  2435,   402],
                [    2,   236,     7,  ...,   860,   103, 22810],
                [    2,     2,     2,  ...,  1911,     6,     8]], device='cuda:1')
        batch_size--2 is: 4
        prompt_length--2 is: 256
        ans--2 is: tensor([[ 370,  197,   28,  ..., 4835,  615,  514],
                [8976,    6,   14,  ..., 4946,    4, 1437],
                [  38,  206,   24,  ...,   38,  206,   24],
                [ 280,   17,   27,  ...,   17,   27,  119]], device='cuda:0')
        valid_ans_len--2 is: tensor([256, 256, 256, 256], device='cuda:0')
        '''

        if self.args.print_answers:
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        # 对每个生成的答案进行处理
        #排除较短（此处设置为有效长度小于1）的answer，余下的answer将被存入out_seq作为最终返回
		
        # yknote zero3的设置下，可能导致out_seq为空！！！
        out_seq = []
        for i in range(batch_size):
            # if valid_ans_len[i] <= 0:  # if the answer is shorter than 1 token, drop it, 改0
            #     continue
            # else: # 对于长度大于1的答案，将其添加到out_seq列表中。
            
            out_seq.append(seq[i:i + 1])
	        # 将out_seq列表中的所有答案序列在批处理维度（也就是第0维）上连接起来，形成一个新的张量。
            # 这个新的张量out_seq就是这个函数的返回值，它包含了所有有效的答案序列。

        # print("len of out_seq--H is", len(out_seq)) #len of out_seq--H

        # # yknote让程序跑通！
        # if len(out_seq) == 4:
        #     out_seq = torch.cat(out_seq, dim=0)  # concate output in the batch dim  #ph3+z3出错！
        #     print("T out_seq--F is", infoTensor(out_seq))
        #     print("xxxx is", out_seq)
        #     return out_seq
        # else:
        #     return None
        #     # print("======================================")
        #     # tmp = torch.rand(4,512).long()
        #     # print("T rand--F is", infoTensor(tmp))===这样还是不行，会出现设备不一致！
        #     # print("tmp is", tmp)
        #     # return tmp

        out_seq = torch.cat(out_seq, dim=0)
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
		生成体验的过程，体验是强化学习中的一个关键概念，用于描述代理(agent)与环境之间的交互。
        生成经验
        :param prompts: prompt input ids，(bs, max_prompt_len)
        :param mask: prompt attention mask, (bs, max_prompt_len)
        :return:
        '''
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)

        #将actor、reference、critic、reward转换为eval模式
        # 给定prompt，生成response text
        # 开启eval模式  
		# 将模型切换到评估模式
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
        ## 获取后，根据解码获得的seq来计算各个奖励和actor与ref对应的logits

        # 调用model.generate()生成序列，由actor模型生成。
        # 输入instruct prompt，由Actor生成seq，上图中红色步骤（1），seq由instruct和response组成
		# 生成序列
        seq = self._generate_sequence(prompts, mask, step)

        # print("seq-1 :", seq)
        # print("T ans-1 :", infoTensor(seq)) #only ph3 x1
        # T ans-1 : _Size([4, 512])_int64_cuda:1_
        ''' 
        seq-1 : tensor([[    2,     2,     2,  ...,    17,    46,     6],
                [    2,     2,     2,  ..., 33837,    35,   653],
                [    2,     2,     2,  ...,    13,    63, 21968],
                [    2,     2,     2,  ...,    17,    27,   119]], device='cuda:0')
        '''

        #将actor、critic转换为train模式，因为后续两者仍需要进行训练
        # 恢复训练模型   
		# 将模型切换回训练模式
        self.train()
        
        # 获取填充符的ID
        pad_token_id = self.tokenizer.pad_token_id
		
        # 创建新的注意力掩码，如果seq中的元素是填充符，那么掩码中的相应位置就是0，否则就是1。 
        attention_mask = seq.not_equal(pad_token_id).long()  #ph3+zero3出错！seq可能为空！
        
        # print("pad_token_id-1 :", pad_token_id)
        # print("attention_mask-1 :", attention_mask)
        # print("T attention_mask :", infoTensor(attention_mask)) #only ph3 x1
        #T attention_mask : _Size([4, 512])_int64_cuda:1_
        '''
        pad_token_id-1 : 2
        attention_mask-1 : tensor([[0, 0, 0,  ..., 1, 1, 1],
                [0, 0, 0,  ..., 1, 1, 1],
                [0, 0, 0,  ..., 1, 1, 1],
                [0, 0, 0,  ..., 1, 1, 1]], device='cuda:0')
        '''

        '''
        actor model是我们想通过强化学习微调的大模型，但是强化学习过程很容易把模型训练“坏”，
        因此需要另外一个不会参数更新的 ref_model来当作标的，别让actor mode跑偏太远。
        我们在训练模式下，将prompt+answer分别输入到actor mode和ref model，
        用KL散度来衡量 ref model和actor mode输出的差别。
        同时将KL散度（衡量数据分布差距大小）纳入损失函数（KL散度本质是纳入到奖励值里边的，奖励值被纳入到了损失函数），
        进而来约束 ref_model和actor mode的输出分布别差距太大。
        具体代码如下：
        '''
        # 对生成的序列（体验）进行评估，得到相关的评估值。
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
			# 使用当前的actor model对序列进行预测，并得到预测的输出。
            # actor model是一个策略网络，用于指导代理做出行动。
            output = self.actor_model(seq, attention_mask=attention_mask)

            # 将seq喂入SFT中得到sft_logits，上图中黑色步骤（5）
			# 使用ref model对序列进行预测，并得到预测的输出。
            # ref model是训练过程中用来比较的基准，它的参数在训练过程中是不变的。
            output_ref = self.ref_model(seq, attention_mask=attention_mask)

            #巨大
            # print("output-1 :", output)
            # print("output_ref-1 :", output_ref)

            # 然后利用reward model和ciric model对输出的prompt+answer进行打分
            # （PPO训练时使用的奖励值并不单单是reward model的输出还要考虑kl散度，后文介绍）：
            # 奖励模型返回的是个字典，key为chosen_end_scores位置存储数据维度为(B,)，表示对于prompt+answer的打分
            '''
            价值函数的forward_value()更具体的细节可见后续详解。
            reward_score取的是answer最后一个token的value
            reward_score.shape: (seq_bs,)
            '''
            # 将生成的序列喂入critic和reward模型中，获得奖励和状态价值
            # 将seq喂入reward模型中打分，得到r(x,  y)，上图绿色步骤（4）
			
			# 使用reward model对序列进行评估，得到序列的奖励值。
            # reward model是用于评估每个行动的好坏的模型，它输出的奖励值将用于指导模型的优化。
            # detach()是将计算的结果与当前的计算图分离，防止在后续的计算中影响梯度的计算。
            reward_score = self.reward_model.forward_value(
                seq, 
				attention_mask,
                prompt_length=self.prompt_length)['chosen_end_scores'].detach()
				
            #巨大
            # print("reward_score-1:", reward_score)
            # print("T reward_score-1 :", infoTensor(reward_score))
            # #only ph3 x1 T T reward_score-1 : _Size([4])_float16_cuda:0_
            '''
            reward_score-1: tensor([ 0.5713,  0.9023, -0.4629,  0.4783], device='cuda:0',
                   dtype=torch.float16)
            '''

            # critic model返回的数据维度为(B,L)，L维度上第i个位置代表从i位置到最后的累积奖励
            # 舍去最后一个位置是因为句子“终止符”无意义

            # critic_model.forward_value(return_value_only=True)将返回shape为(seq_bs, max_seq_len)的序列各token的value
            # 将seq喂入critic，获得critic的value，上图蓝色步骤（3）
			
			# 使用critic model对序列进行评估，得到序列的价值估计。
            # critic model是用于评估状态的价值模型，它输出的价值估计将用于指导模型的优化。
            # [:, :-1]是取除了最后一列之外的所有列，这是因为在计算回报时，通常会忽略最后一个状态的价值估计。
            values = self.critic_model.forward_value(
                seq, attention_mask, return_value_only=True).detach()[:, :-1]

            # print("values-1 :", values)
            # print("T values-1 :", infoTensor(values))
            # #T values-1 : _Size([4, 511])_float16_cuda:1_  only ph3 x1

        '''知识补充:
        reward_model 与 critic_model 的异同，查看思维导图。
        '''
		
		# logits是actor_model的输出，代表了在每个可能的输出位置，每种可能的词或字符的原始未归一化的分数。
        # (seq_bs, max_seq_len, vocab_size)
        logits = output.logits
        # print("logits-1 :", logits)

        # logits_ref是ref_model的输出，代表了在每个可能的输出位置，每种可能的词或字符的原始未归一化的分数。
        # (seq_bs, max_seq_len, vocab_size)
        logits_ref = output_ref.logits
        # print("logits_ref-1 :", logits_ref)


        # print("T logits-1 :", infoTensor(logits))  #only ph3 x1
        # print("T logits_ref-1 :", infoTensor(logits_ref)) #only ph3 x1
        # T logits-1 : _Size([4, 512, 50272])_float16_cuda:1_
        # T logits_ref-1 : _Size([4, 512, 50272])_float16_cuda:1_
        '''
        values-1 : tensor([[-0.4238, -0.4238, -0.4238,  ..., -0.6475, -0.5742, -0.3826],
                ...
                [ 0.7188,  0.7188,  0.7188,  ...,  0.5762,  0.3242,  0.1738]],
               device='cuda:1', dtype=torch.float16)
        logits-1 : tensor([[[ -7.1914,  -7.1875,   2.5566,  ...,  -7.1758,  -7.1719,  -7.1406],
                ...
                 [ -6.6836,  -6.6602,   5.5117,  ...,  -6.5547,  -6.6680,  -6.6484]]],
               device='cuda:1', dtype=torch.float16)
        logits_ref-1 : tensor([[[ -7.1914,  -7.1875,   2.5566,  ...,  -7.1758,  -7.1719,  -7.1406],
                ...
                 [ -6.6836,  -6.6602,   5.5117,  ...,  -6.5547,  -6.6680,  -6.6484]]],
               device='cuda:1', dtype=torch.float16)
        
        '''

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
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:,1:]),
            'value': values, # critic model产生的状态值，它表示了在各种状态下agent的预期未来回报。
            'rewards': reward_score, # 由reward model产生的即时奖励评分，评估了生成的答案序列的质量。
            'input_ids': seq, # 生成的答案序列
            "attention_mask": attention_mask # 掩码，标识出了在答案序列中哪些是有效的token，哪些是填充的token。
        }

    ## action_mask = attention_mask[:, 1:]
    ## reward_score shape = [bs]
    ## prompt means the inputs for model.generate() method, so the input length is aligned

    ## 获取上述critic，reward，actor logits 和ref logits后计算rewards，
    # 最终rewards为 actor 和 ref 的 logits 差加上 clamp后的reward_scores

    # 如何计算Advantage？
    def compute_rewards(self,
                        prompts,
                        log_probs, # 每个行为的对数概率
                        ref_log_probs, # 参考行为的对数概率
                        reward_score, # 奖励模型给出的奖励
                        action_mask):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)

        # 计算kl散度，log_probs里边存的数字经过log变化了，因此减法就对应除法
        """
        计算实际rewards，涉及（旧）策略与SFT的KL散度惩罚、RM的reward
        计算经验采样时actor与SFT的KL散度惩罚
        """
        # 计算KL散度的估计，KL散度用于度量两个概率分布之间的相似性，因此这个估计值代表了actor模型和参考模型生成行为的相似性。
        # 它在更新模型参数时，可以作为行为奖励的一部分
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        # print("kl_divergence_estimate is:", kl_divergence_estimate)
        # print("T kl_divergence_estimate:", infoTensor(kl_divergence_estimate))
        # T kl_divergence_estimate: _Size([4, 511])_float16_cuda:0_    only ph3
        '''
        kl_divergence_estimate is: tensor([[-3.9053e-04, -3.9053e-04, -3.9053e-04,  ..., -6.7115e-05,
         -1.0526e-04, -1.5914e-05],
        ...
        [-0.0000e+00, -0.0000e+00, -0.0000e+00,  ..., -2.2435e-04,
         -1.8435e-03,  5.3644e-07]], device='cuda:0', dtype=torch.float16)
        '''

        # PPO训练时候的奖励值综合考虑KL散度和reward模型的输出，只考虑answer部分的KL散度，将reward
        # model的输出加到KL散度L维度的最后一个位置上，得到最终的奖励值，代码如下：
        rewards = kl_divergence_estimate

        # 只考虑answer部分的奖励，不考虑prompt
        """
        找到answer的起始start：即prompt的最后1个token位置
        比如prompts长度为256，answer的起始则为256-1=255
        """
        # 状态s_1在prompt最后一个token，动作a_1表示预测response的第一个token
		# 找到每个对话的开始和结束的位置
        start = prompts.shape[1] - 1

        # 不考虑padding部分
        '''
		ends为batch中各个数据的最后1个有效token的index，
		每个数据的最末有效token位置很大可能是不一样的，
		因此ends是个数组
		'''
        ends = start + action_mask[:, start:].sum(1) + 1

        ## rewards_scores 仅为最后一个非pad token的值，shape 为 bs * 1
        ## values 是 return_values_only 得到的，为每一个token对应的rewards 分数

        # 将RM得到的奖励值限定在一定范围，默认为(-5,5)
		# 将奖励分数进行了剪裁，也就是将奖励分数限制在一个范围内。
        # 范围的上下限由self.clip_reward_value设置，这是防止奖励分数过大或过小，影响模型的学习。
        reward_clip = torch.clamp(reward_score, 
		                          -self.clip_reward_value,
                                  self.clip_reward_value)
								  
        batch_size = log_probs.shape[0]
        # print("batch_size is:", batch_size)
        # print("reward_clip is:", reward_clip)
        # print("T reward_clip--A:", infoTensor(reward_clip))
        '''
        T reward_clip--A: _Size([4])_float16_cuda:0_
        
        batch_size is: 4
        reward_clip is: tensor([ 1.5645, -0.7383, -0.5581, -2.2441], device='cuda:0',
               dtype=torch.float16)
        '''

        # 在L维度上，每个位置都有KL散度，但是只在最后一个位置加上奖励值
        '''
        因为batch中每个数据的最末有效token位置很可能不一样，
		所以无法通过矩阵来并行，需要使用for循环逐个数据处理
		'''
        # 遍历每一个对话
        for j in range(batch_size):
            """
            KL_reward = KL + reward
            加和只在最末有效token上进行
            """
            # 在最后一个token加reward_score
	        # 将剪裁后的奖励分数加到了每个对话的最后一个行为上
            # rewards[j, start:ends[j]]选出了第j个对话的所有行为，[-1]选出了最后一个行为。
            # 每个对话的最后一个行为将得到额外的奖励。
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        """返回KL rewards"""
        # print("return rewards is:", rewards)
        # print("T reward_clip--B:", infoTensor(rewards))
        '''
        
        T reward_clip--B: _Size([4, 511])_float16_cuda:1_

        return rewards is: tensor([[-3.9053e-04, -3.9053e-04, -3.9053e-04,  ..., -6.7115e-05,
         -1.0526e-04,  1.5645e+00],
        ...
        [-0.0000e+00, -0.0000e+00, -0.0000e+00,  ..., -2.2435e-04,
         -1.8435e-03, -2.2441e+00]], device='cuda:0', dtype=torch.float16)
        '''
        return rewards
        ## 用计算好的 rewards来更新actor model，values来更新critic model


    '''
    3.3.5.2 PPO训练
    012.png
    1次PPO训练由train_rlhf()方法进行管理，其内部主要实现了：
    013.png
    具体代码可见下方，为保证阅读的流畅性，我对其中的部分代码进行了调整，
    使得相应的函数代码衔接在其调用后方，便于具体对照其传参，从而辨析传入的新旧策略、新旧价值估计等：
    '''
    def train_rlhf(self, inputs):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)

        # 使用1个ppo_batch的经验数据，执行1次rlhf训练迭代

        # train the rlhf mode here
        ### process the old outputs
        # prompt input ids
        # 当前RLHF轮次最初采样的经验池中采样一批数据
        # instruct prompt
		# 一个batch的提问/上下文序列
		# 输入的prompt（例如in-context exemplar + query）
        prompts = inputs['prompts']  

        # （旧）策略
        # actor模型生成response对应的action_logist
		# 当前actor模型对下一步动作的log概率
		# 根据prompt，actor模型生成的文本的概率
        log_probs = inputs['logprobs'] 

        # SFT策略
        # SFT模型生成response对应的sft_logits
		# 根据prompt，reference生成模型的文本的概率
		# ref模型对下一步动作的log概率
        ref_log_probs = inputs['ref_logprobs']

        # RM奖励
        # reward模型预测的奖励r(x, y)
		# reward模型为生成的序列给出的奖励
		# 根据prompt生成的seq，reward模型得到的奖励
        reward_score = inputs['rewards']  

        # （旧）价值估计
        # critic模型预测的奖励
		# 根据prompt生成的seq，critic模型得到的状态价值函数值
		# critic模型预测的未来奖励的价值
        values = inputs['value']  
		
        # 哪些序列元素是填充的
		# actor生成的文本的attention mask
        attention_mask = inputs['attention_mask']  

        # seq input ids
		# actor模型生成的完整序列
		# 根据prompt，actor生成的文本
        seq = inputs['input_ids'] 

        """
        获取prompts的最后1个位置作为start
        比如prompt_len为256，start则为 256-1=255
        这个start主要是用于取出经验数据中的“非prompt”部分（也即“answer+padding”部分）
        """
        # 开始处理序列的位置
		# 记prompt文本最后一个位置
        start = prompts.size()[-1] - 1   

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
	    # 去掉了每个序列的第一个元素
        # 原因：在RLHF中，每个序列的第一个元素通常是一个特殊的起始标记，如[CLS]或<s>，并不对应于实际的动作。
        action_mask = attention_mask[:, 1:]

        # print("prompts is:", prompts)
        # print("log_probs is:", log_probs)
        # print("ref_log_probs is:", ref_log_probs)
        # print("reward_score is:", reward_score)
        # print("values is:", values)
        # print("attention_mask is:", attention_mask)
        # print("seq is:", seq)
        # print("start is:", start)
        # print("action_mask is:", action_mask)

        # print("T prompts:", infoTensor(prompts))
        # print("T log_probs:", infoTensor(log_probs))
        # print("T ref_log_probs:", infoTensor(ref_log_probs))
        # print("T reward_score:", infoTensor(reward_score))
        # print("T values:", infoTensor(values))
        # print("T attention_mask:", infoTensor(attention_mask))
        # print("T seq:", infoTensor(seq))
        # print("T action_mask:", infoTensor(action_mask))
        '''
        T prompts: _Size([4, 256])_int64_cuda:0_
        T log_probs: _Size([4, 511])_float16_cuda:0_
        T ref_log_probs: _Size([4, 511])_float16_cuda:0_
        T reward_score: _Size([4])_float16_cuda:0_
        T values: _Size([4, 511])_float16_cuda:0_
        T attention_mask: _Size([4, 512])_int64_cuda:0_
        T seq: _Size([4, 512])_int64_cuda:0_
        T action_mask: _Size([4, 511])_int64_cuda:0_
        '''

        '''
        prompts is: tensor([[    2,     2,     2,  ..., 50118, 46184,    35],
                ...
                [    2,     2,     2,  ..., 50118, 46184,    35]], device='cuda:1')
        log_probs is: tensor([[-5.8633e+00, -5.8633e+00, -5.8633e+00,  ..., -1.2253e-02,
                 -1.6815e-02, -3.3474e-03],
                ...
                [-2.0078e+00, -2.0078e+00, -2.0078e+00,  ..., -1.9516e-02,
                 -2.6443e-02, -6.1607e-03]], device='cuda:1', dtype=torch.float16)
        ref_log_probs is: tensor([[-5.8633e+00, -5.8633e+00, -5.8633e+00,  ..., -1.3519e-02,
                 -1.8341e-02, -3.7823e-03],
                ...
                [-2.0391e+00, -2.0391e+00, -2.0391e+00,  ..., -2.1149e-02,
                 -2.8412e-02, -6.6376e-03]], device='cuda:1', dtype=torch.float16)
        reward_score is: tensor([-0.4282,  1.4141, -0.3965,  1.2178], device='cuda:1',
               dtype=torch.float16)
        values is: tensor([[-0.4321, -0.4321, -0.4321,  ..., -0.4736, -0.4829, -0.3918],
                ...
                [ 1.5361,  1.5361,  1.5361,  ...,  1.2910,  1.3447,  1.2559]],
               device='cuda:1', dtype=torch.float16)
        attention_mask is: tensor([[0, 0, 0,  ..., 1, 1, 1],
                ...
                [0, 0, 0,  ..., 1, 1, 1]], device='cuda:1')
        seq is: tensor([[   2,    2,    2,  ...,   47,   64,   67],
                ...
                [   2,    2,    2,  ...,    7, 7142,   24]], device='cuda:1')
        start is: 255
        action_mask is: tensor([[0, 0, 0,  ..., 1, 1, 1],
                ...
                [0, 0, 0,  ..., 1, 1, 1]], device='cuda:1')
        '''

        # 经验数据中的价值估计为“旧”价值估计
        ### 根据经验数据，接下来计算相应的reward和advantage
		# 价值函数的预测结果
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
			
			# 使用KL散度，参考模型和行动模型的日志概率，和奖励模型得到的奖励值来计算每个动作的奖励
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)

            ends = start + action_mask[:, start:].sum(1) + 1

            # print("old_rewards is:", old_rewards)
            # print("ends is:", ends)

            # print("T old_rewards:", infoTensor(old_rewards))
            # print("T ends:", infoTensor(ends))
            '''
            T old_rewards: _Size([4, 511])_float16_cuda:1_
            T ends: _Size([4])_int64_cuda:1_

            old_rewards is: tensor([[-3.9053e-04, -3.9053e-04, -3.9053e-04,  ..., -6.7115e-05,
                     -1.0526e-04,  1.5645e+00],
                    [-1.1721e-03, -1.1721e-03, -1.1721e-03,  ..., -3.3557e-05,
                     -1.8299e-05, -7.3828e-01],
                    [-1.5621e-03, -1.5621e-03, -1.9526e-04,  ..., -7.6294e-05,
                     -6.7949e-06, -5.5811e-01],
                    [-0.0000e+00, -0.0000e+00, -0.0000e+00,  ..., -2.2435e-04,
                     -1.8435e-03, -2.2441e+00]], device='cuda:0', dtype=torch.float16)
            ends is: tensor([512, 512, 512, 512], device='cuda:0')
            '''

            # yknote
            # we need to zero out the reward and value after the end of the conversation
            # otherwise the advantage/return will be wrong
            for i in range(old_rewards.shape[0]):
                ## 之前生成_generate_sequence时并没有过滤掉pad_token，所以要把pad_token处的rewards 记 0
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

            ## old_values 为 critic model计算的每个token 对应的rewards分数
            ## old_rewards为actor 和 ref 做logits差后加上 reward model 的最后一个token 的reward分数
			
			# advantages(优势)表示在给定状态下，一个动作比策略平均水平好多少。
            # returns(回报)则是在时间t，执行一个动作后预期能得到的总奖励。
            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

            # print("advantages is:", advantages)
            # print("returns is:", returns)
            # print("T advantages:", infoTensor(advantages))
            # print("T returns:", infoTensor(returns))
            '''
            T advantages: _Size([4, 256])_float16_cuda:1_
            T returns: _Size([4, 256])_float16_cuda:1_

            advantages is: tensor([[-0.1390,  0.0554, -0.3184,  ...,  0.3071,  0.2463,  0.0576],
                    [ 0.2349,  0.5220,  0.0591,  ...,  0.3701,  0.2075,  0.1572],
                    [ 0.3958,  0.3511,  0.6201,  ...,  0.0502,  0.1464,  0.0811],
                    [ 0.2108,  0.1600,  0.0022,  ..., -0.2986, -0.0490,  0.1067]],
                   device='cuda:1', dtype=torch.float16)
            returns is: tensor([[-0.5776, -0.5752, -0.5908,  ..., -0.3403, -0.3279, -0.3250],
                    [-0.1965, -0.1704, -0.1674,  ..., -0.4312, -0.4209, -0.4131],
                    [ 0.7368,  0.7544,  0.7856,  ...,  1.2734,  1.2812,  1.2852],
                    [ 0.5503,  0.5581,  0.5586,  ...,  0.2776,  0.2751,  0.2805]],
            '''

        ### process the new outputs
        # ###计算actor损失并更新
        # 下面则是获得生成部分seq的奖励等信息
        ### 根据经验数据以及得到的advatage，下面开始获得一系列的loss
        batch = {'input_ids': seq, "attention_mask": attention_mask}
		
        # print("T batch['input_ids']:", infoTensor(batch['input_ids']))
        # print("T batch['attention_mask']:", infoTensor(batch['attention_mask']))
        # print("batch is:", batch)
        '''
        batch is: {'input_ids': tensor([[    2,     2,     2,  ...,    64,    67, 10397],
        ...
        [    2,     2,     2,  ...,    10,   357,  4885]], device='cuda:0'), 
        'attention_mask': tensor([[0, 0, 0,  ..., 1, 1, 1],
        ...
        [0, 0, 0,  ..., 1, 1, 1]], device='cuda:0')}
        
        T batch['input_ids']: _Size([4, 512])_int64_cuda:0_
        T batch['attention_mask']: _Size([4, 512])_int64_cuda:0_
        '''

        #将seq经验数据输入至actor，进行自回归预测
        # 获得seq的的概率

        # 将这一批经验数据的seq（instruct prompt+response）再一次喂入actor得到logits
        # 因为现在是在更新actor和critic，而经验数据所采用的actor和critic早已经是之前的了，所以
        # 现在正在更新的actor和critic与当时进行经验采样时的actor、critic的参数已经有差异了；
        # 所以需要重新获得当前最新的actor输出的logits
        # 上图中棕色步骤（3）
		
		# 使用这个batch通过actor_model生成新的预测，这个预测包含了所有可能动作的概率分布actor_prob
        actor_prob = self.actor_model(**batch, use_cache=False).logits

        #取出probs，此处为新策略
		# 根据生成的动作序列seq，从actor_prob中提取出实际采用的动作对应的日志概率actor_log_prob
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])


        # print("actor_prob is:", actor_prob)
        # print("actor_log_prob is:", actor_log_prob)
        # print("T actor_prob:", infoTensor(actor_prob))
        # print("T actor_log_prob:", infoTensor(actor_log_prob))
        '''
        T actor_prob: _Size([4, 512, 50272])_float16_cuda:0_
        T actor_log_prob: _Size([4, 511])_float16_cuda:0_

                actor_prob is: tensor([[[-6.6680e+00, -6.6641e+00,  2.8535e+00,  ..., -6.6484e+00,
                  -6.6602e+00, -6.6328e+00],
        ....
                 [ 1.3191e-02,  2.1805e-02,  3.0352e+00,  ...,  7.7942e-02,
                   2.4506e-02, -2.8662e-01]]], device='cuda:0', dtype=torch.float16,
               grad_fn=<UnsafeViewBackward0>)
        actor_log_prob is: actor_loss is: tensor([[-5.9297e+00, -5.9297e+00, -5.9297e+00,  ..., -5.8479e-03,
                 -1.4099e-02, -3.5980e-02],
        ...
                 -2.6123e-02, -1.9791e-02]], device='cuda:0', dtype=torch.float16,
               grad_fn=<SqueezeBackward1>)
        '''

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
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:], # 新的日志概率
                                        log_probs[:, start:], # 旧的日志概率
                                        advantages, # 优势
                                        action_mask[:, start:])


        #actor反向传播、更新参数
        # 更新actor参数
        # 更新actor模型参数
        self.actor_model.backward(actor_loss)
        # print("actor_loss is:", actor_loss)
        # actor_loss is: tensor(0.0085, device='cuda:0', dtype=torch.float16, grad_fn=<DivBackward0>)

        if not self.args.align_overflow:
            self.actor_model.step()

        #计算critic损失并更新################################################
        #将seq经验数据输入至critic，预测得到新价值估计
        # 获得seq的critic得分

        # 经验数据中的seq（instruct prompt+response）再一次喂入critic得到value
        # 同理，由于当前的critic和当初进行经验数据采样时的critic相差很远；所以需要重新获得value
        # 上图中黑色步骤（5）
		# 预测出每个动作的价值估计
        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]

        # print("value is:", value)
        # print("T value:", infoTensor(value))
        '''
        T value: _Size([4, 511])_float16_cuda:1_
        value is: tensor([[-0.3354, -0.3354, -0.3354,  ...,  0.2708,  0.1158,  0.2218],
        ...
        [-0.2240, -0.2240, -0.2240,  ...,  1.1074,  1.1123,  1.0693]],
            device='cuda:0', dtype=torch.float16, grad_fn=<SliceBackward0>)
        '''

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

        # print("critic_loss is:", critic_loss)
        # critic_loss is: tensor(0.0110, device='cuda:0', dtype=torch.float16, grad_fn=<DivBackward0>)

        # critic反向传播、更新参数
        # 更新Critic模型参数
        # 更新critic参数
        self.critic_model.backward(critic_loss)

        if self.args.align_overflow:
            actor_overflow = self.actor_model.optimizer.check_overflow(
                external=True)
            critic_overflow = self.critic_model.optimizer.check_overflow(
                external=True)

            print("actor_overflow is:", actor_overflow)
            print("critic_overflow is:", critic_overflow)

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
        '''
        actor model : 试图最大化预期回报
        critic model : 试图尽可能准确地估计每个动作的价值
        '''
        return actor_loss, critic_loss

        ### process the new outputs

    '''
    batch = {'input_ids': seq, "attention_mask": attention_mask}

    ## 训练时外部还有 ppo_epoch的循环，但默认值为1，所以这两个log都是同一个模型（有无dropout）计算出来的
    actor_prob = self.actor_model(**batch, use_cache=False).logits
    actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])
    actor_loss = 
    self.actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages, action_mask[:, start:])
    '''

    '''
    以上过程，我们已经拿到了PPO训练所需要的advantage以及actor model的输出，我先现在可以对actor model进行训练啦。
    具体代码如下。
    logprobs和old_logprobs这两个参数分别是“老actor（n个epoch才会更新一次）”和新actor（每个batch都会更新它）”在正确单词上出处的概率，
    这块时PPO import sampling相关的知识，就不在这重复介绍了，不明白的同学补习一下哈。借用一下李宏毅老师的PPO公式：
    图！

    '''

    # Clipped Surrogate Objective 033.png  对应为更新actor的loss
    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        '''
		定义了PPO算法中的策略梯度损失函数，它是计算actor model的损失函数的一部分。
           PPO的主要思想，即通过限制新旧策略之间的差异来稳定学习过程，同时仍然允许策略改进以获得更好的性能。
		'''
        #"""计算actor的损失"""

        ## policy gradient loss
        # logprobs, old_logprobs都是经过log变化的单词概率，这里带着log做减法就相当于在做概率除法

        # 重要性采样权重计算：ratio = exp(log(new)-log(old))
        log_ratio = (logprobs - old_logprobs) * mask

        # 指数操作去掉log
        ratio = torch.exp(log_ratio)

        # 计算策略梯度损失的2个情况：加权优势 与 裁剪加权优势
		# 在PPO中，策略损失是由两部分组成的：一部分是按照策略概率比例（即新旧策略的比例）加权的期望优势函数
        pg_loss1 = -advantages * ratio
        # 另一部分是将这个比例裁剪到特定范围后的期望优势函数
        # PPO的目标：在尝试优化策略以获得更好的预期回报的同时，也限制新旧策略之间的差异。
        # 裁剪比例就是实现这一目标的一种方式，它防止新策略偏离旧策略太远。
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)

        # 从2个情况中选择损失较大者作为真正的损失，
		# 并且基于ppo_batch内所有数据的所有有效时间步计算平均损失值
		# 下面公式解析：
        # ① torch.max(pg_loss1, pg_loss2)：这两部分中的较大值被用来计算策略损失
        # ② 乘以mask，这是为了确保只有有效的时间步（不包括填充的时间步）被用来计算损失
        # ③ 损失是所有有效时间步上损失的总和，除以有效时间步的总数，得到一个平均损失。
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    #同样的，我们也要对critic model进行训练，更新，loss就是mse loss。
    def critic_loss_fn(self, values, old_values, returns, mask):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)

        # 计算价值损失

        ## value loss
        ## 用“老critic model”的输出约束“新critic model”不要步子太大，裁剪一下
        '''
        至此，我们的RLHF训练流程就结束了。
        第二部分开头我们说过，共涉及actor model， ref_model，reward model和critic model这四个模型，
        其实更新参数的模型只有actor model和critic model。

        裁剪当前新values，使得其不至于太偏离经验采样阶段的旧values
        '''
        # 价值预测values被裁剪到一个由old_values确定的范围内，该范围由self.cliprange_value控制。
        # 这是为了防止新旧价值预测之间的差异过大，以稳定学习过程。
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
        )

        # 计算当前values与回报的L2 Loss
		# 计算价值预测values与实际回报returns之间的均方误差vf_loss1
        vf_loss1 = (values - returns)**2

        # 计算裁剪后的当前values与回报的L2 Loss
		# 计算裁剪后的价值预测values_clipped与实际回报returns之间的均方误差vf_loss2
        vf_loss2 = (values_clipped - returns)**2

        """
        选择损失较大者作为真正的损失，
        并且基于ppo_batch内所有数据的所有有效时间步计算平均损失值，
        此外critic损失项的系数为0.5。
        """
        # 计算平均损失
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()

        '''补充:
        这个损失函数的目标是减小价值函数预测的误差，同时也限制新旧价值预测之间的差异。
        价值函数预测的准确性对于策略更新非常重要，因为它用于计算优势估计，而优势估计是用来更新策略。
        '''
        return vf_loss

    '''
    接下来的内容就是PPO的训练过程的比较核心的内容了，目标是计算PPO更新公示里边的advantage，
    具体公式如下，V就是critic model的输出。如果原理不懂建议先到这个链接
    https://link.zhihu.com/?target=https%3A//huggingface.co/blog/deep-rl-a2c
    看看。我直接在代码中给注释了。
    图！

    图片出处：https://huggingface.co/blog/deep-rl-a2c
    '''

    ## values 为 更新参数前critic model对actor 解码得到的seq计算得到的各个token评分，shape 为 [bs * seq]
    ## rewards is the output from compute_rewards function, with shape [bs * seq]
    ## 公式 035.png

    def get_advantages_and_returns(self, values, rewards, start):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        '''定义了如何计算Generalized Advantage Estimation (GAE) 和 returns（即每个时间步的累积奖励），
           这两个量都用于PPO (Proximal Policy Optimization)训练过程。
        '''

        # values（B，L） critic model输出
        # rewards（B，）reward model输出
        # start answer开始的位置
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134

        # 计算优势与回报
        # 实现基本与上述公式相同
        lastgaelam = 0 # 初始化最后一步的GAE为0
        advantages_reversed = [] # 用于存储计算出的逆序的GAE
        length = rewards.size()[-1] # 在一个序列中的时间步长


        # 计算每个时刻（序列位置）的critic model预测误差
        # 反向遍历计算各个时间步的优势advantage
        # 遍历所有时间步（从后往前）
        for t in reversed(range(start, length)):
            # 获取下个时间步的价值估计V_{old}(s_{t+1})
			# 如果不是最后一个时间步，那么nextvalues就是下一步的价值预测，否则设为0。
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0

            # critic model预测的是t到到最后一个时刻的奖励和，所以变化量delta可以用如下公式表示
			# 计算误差delta，即当前步的奖励加上下一步的折扣后的价值预测，再减去当前步的价值预测。
            """计算单步TD-error"""
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]

            """累计优势"""
            # self.gamma=1，self.lam=0.95是衰减因子，表示之前计算的delta对现在影响越来越小
			# 计算当前步的GAE，等于误差加上折扣的、衰减的上一步的GAE
            lastgaelam = delta + self.gamma * self.lam * lastgaelam

            # """存储各个时间步的优势"""
			# 将计算出的GAE添加到逆序的GAE列表中
            advantages_reversed.append(lastgaelam)

        #对逆序的优势列表进行正序处理，得到正常时间步排列的优势
		# 将逆序的GAE列表反向，得到正向的GAE
        advantages = torch.stack(advantages_reversed[::-1], dim=1)  # 再反转
		
        #太大
        # print("advantages_reversed--1 is:", advantages_reversed)
        # print("advantages--1 is:", advantages)

        # 后续用来更新critic model用
        """
        return_t = adv_t + v(s_t)
        由优势计算得到回报
        """
        # 将其与价值预测相加，得到返回值
		# adv(t) + value(t+1)更合理些
        returns = advantages + values[:, start:]  

        # 返回优势与回报
        return advantages.detach(), returns

    def _validate_training_mode(self):
        '''确保actor_model和critic_model都处于训练模式，如果模型正在训练，该属性将为True。'''
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        '''确保actor_model，critic_model，ref_model和reward_model都不在训练模式，即在评估模式。
           如果模型正在评估，该属性将为False。'''
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        #gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        # gd.debuginfo(prj='ds-chat', info="start actor_model.train")
        self.actor_model.train()
        # gd.debuginfo(prj='ds-chat', info="end actor_model.train")
        #
        gd.debuginfo(prj='ds-chat', info="-start critic_model.train")
        self.critic_model.train()
        gd.debuginfo(prj='ds-chat', info="end critic_model.train")

    def eval(self):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        # gd.debuginfo(prj='ds-chat', info="start actor_model.eval")
        self.actor_model.eval()
        # gd.debuginfo(prj='ds-chat', info="end actor_model.eval")
        #
        # gd.debuginfo(prj='ds-chat', info="start critic_model.eval")
        self.critic_model.eval()
        # gd.debuginfo(prj='ds-chat', info="end critic_model.eval")
        #
        # gd.debuginfo(prj='ds-chat', info="start reward_model.eval")
        self.reward_model.eval()
        # gd.debuginfo(prj='ds-chat', info="end reward_model.eval")
        #
        # gd.debuginfo(prj='ds-chat', info="start ref_model.eval ")
        self.ref_model.eval()
        # gd.debuginfo(prj='ds-chat', info="end ref_model.eval")

    def dump_model_norms(self, tag):
        '''计算并打印每个模型（actor_model，ref_model，critic_model和reward_model）的参数范数
           参数范数常常被用作评估模型复杂度的指标，也可以用于诊断和监视训练过程中的梯度爆炸或消失问题。'''
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        # 在所有的分布式工作进程上打印每个模型的范数，
        # 这对于在使用分布式训练时监视模型复杂度和稳定性很有帮助
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
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        gd.debuginfo(prj='ds-chat', info=self.__class__.__name__)

        """
        1个ppo_batch的无监督训练
        :param inputs: dict：input_ids, attention_mask, labels
        :param unsup_coef: 无监督损失系数

        确保actor处于训练模式，否则将返回报错
        """

        # Train the unsupervised model here
        # 确保模型处于训练模式，而不是评估模式。
        self._validate_training_mode()

        #actor进行常规的CausalLM训练
		# 前向传播
        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss

        # 反向传播、更新参数
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
