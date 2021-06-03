from itertools import compress
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F
from utils import decision
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

class ConstrainedGenText:
    """
    Constraint就是一开始的text的构成
    """
    def __init__(self, text, tokenized_text_input_ids, hard_constraint_mask):
        """
        text 是原text，且所有字母小写
        tokenized_text 是经过PLM tokenize之后的，注意是直接tokenize之后，所以包括input_ids、token_type_ids、attention_mask，且dim为2，第一维恒为1
        hard_constraint 是一个sequence的list
        """
        self.text = text.lower()
        # self.mlm_tokenizer = mlm_tokenizer
        # tokenized_text与进行操作的text一定要一样
        # self.tokenized_text = self.mlm_tokenizer(text, add_special_tokens=False, return_tensors="pt")
        self.tokenized_text = tokenized_text
        self.hard_constraint_mask = hard_constraint_mask


    def update_hard_constraint_mask(self, position, action):
        """
        根据操作和进行操作的位置，更新Text的hard_constraint_mask
        Args:
            position ([int]): 进行操作的位置
            action ([str]): 进行的操作名，只可能为"delete" / "insert"
        """
        if action == "insert":
            self.hard_constraint_mask.insert(position, 1)
        elif action == "delete":
            self.hard_constraint_mask.pop(position)
        else:
            raise ValueError("action must be delete or insert")


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

class MHSG:
    def __init__(self, MLM_model, MLM_tokenizer, LM_model, LM_tokenizer):
        """
        MLM_model 是 transformers 库中的进行 Masked Language Modeling 的模型对象
        LM_model 是 transformers 库中的进行 Language Modeling的模型对象
        """
        self.MLM_model = MLM_model
        self.MLM_tokenizer = MLM_tokenizer
        self.LM_model = LM_model
        self.LM_tokenizer = LM_tokenizer
        self.replace_prior = 1/3
        self.insert_prior = 1/3
        self.delete_prior = 1/3
    def get_position(self, input_text:ConstrainedGenText, last_position:int):
        """
        得到当前操作进行的位置
        Args:
            input_text (ConstrainedGenText): 输入的text
            last_position (int): 上一次进行操作的位置
        """
        aviable_indices = list(compress(range(len(input_text.hard_constraint_mask)), input_text.hard_constraint_mask))
        cur_position = random.choice(aviable_indices)
        assert cur_position < len(input_text.tokenized_text.input_ids[0])
        return cur_position
    
    def get_whole_text_prod(self, text:str):
        self.LM_model.eval()
        with torch.no_grad():
            input_ids = self.LM_tokenizer(text, return_tensors='pt').input_ids
            target_ids = input_ids.clone()
            text_ppl = torch.exp(self.LM_model(input_ids, labels=target_ids)[0]).item()
            text_prob = text_ppl ** (-input_ids.size(0))
        return text_prob
    
    def get_MLM_prod(self, text:ConstrainedGenText, position:int, top_p=0, top_k=0):
        self.MLM_model.eval()
        with torch.no_grad():
            logits = self.MLM_model(**text.tokenized_text).logits[0][position]

            if top_p != 0 or top_k != 0: # sampling
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                proposal_prod = F.softmax(logits, dim=-1)
                proposal_token = proposal_prod.multinomial(num_samples=1)
            else: # greedy search
                proposal_prod = F.softmax(logits, dim=-1)
                proposal_token = proposal_prod.argmax(-1)
            return proposal_prod, proposal_token

    def replace(self, input_text, position, top_p=0, top_k=0):
        """
        进行替换操作，
        Args:
            input_text (ConstrainedGenText): 进行操作的原始Text
            position (int): 进行操作的位置
        """
        self.LM_model.eval()
        self.MLM_model.eval()
        with torch.no_grad():
            logger.info("=" * 20 + " Replace " + "=" * 20)
            logger.info("Old Text is : {}".format(input_text.text))
            old_text_prob = self.get_whole_text_prod(input_text.text)
            proposal_prod, proposal_token = self.get_MLM_prod(input_text, position, top_p, top_k)
            if proposal_token in self.MLM_tokenizer.all_special_ids: # 如果预测的token是special token，直接返回input_text
                logger.info("The proposal token is a special token, so return the original text.")
                return copy.deepcopy(input_text)

            proposal_token_prod = proposal_prod[proposal_token] #预测得用于替换的token的条件概率
            old_token_prod = proposal_prod[input_text.tokenized_text.input_ids[0][position]] #原来的token的条件概率

            proposal_tokenized_text = copy.deepcopy(input_text.tokenized_text) # token_type_ids和attention_mask应该是不变的
            # 第一个index为0，因为batch_size为1
            proposal_tokenized_text.input_ids[0][position] = proposal_token
            proposal_text = self.MLM_tokenizer.batch_decode(proposal_tokenized_text.input_ids, skip_special_tokens = True)[0]
            proposal_text_prob = self.get_whole_text_prod(proposal_text)

            A_replace = min(1, proposal_text_prob * old_token_prod / (old_text_prob * proposal_token_prod))
            
            logger.info("Proposal Text is : {}".format(proposal_text))
            logger.info("Accept rate: {ar} ; Proposal text prob: {pp} ; Old token prob: {otp} ; Old text prob: {op} ; Proposal token prob: {ptp} ".format(ar=A_replace, pp=proposal_text_prob, otp=old_token_prod, op=old_text_prob, ptp=proposal_token_prod))

            if decision(A_replace):
                accept_text_hard_constraint_mask = copy.deepcopy(input_text.hard_constraint_mask)
                accept_text = ConstrainedGenText(proposal_text, proposal_tokenized_text, accept_text_hard_constraint_mask)
                return accept_text
            else:
                return copy.deepcopy(input_text)

    def insert(self, input_text, position, top_p=0, top_k=0):
        """
        进行插入操作，先在position位置插入一个<MASK>，再将改<MASK>替换成其他单词
        Args:
            input_text (ConstrainedGenText): 进行操作的原始Text
            position (int): 进行操作的位置
        """
        self.LM_model.eval()
        self.MLM_model.eval()
        with torch.no_grad():
            logger.info("=" * 20 + " Insert " + "=" * 20)
            logger.info("Old Text is : {}".format(input_text.text))
            old_text_prob = self.get_whole_text_prod(input_text.text)
            
            tmp_text = copy.deepcopy(input_text)
            input_text_input_ids = input_text.tokenized_text.input_ids[0]
            # 在position位置插入一个 <mask> , 103是<mask>对应的id
            tmp_text.tokenized_text.input_ids = torch.cat(input_text_input_ids[:position], 103, input_text_input_ids[position:]).unsqueeze(0)
            tmp_text.text = self.MLM_tokenizer.batch_decode(tmp_text.tokenized_text.input_ids, skip_special_tokens = True)[0]
            tmp_text.update_hard_constraint_mask(position, "insert")
            logger.info("Insert a <MASK>, and then replace it. ")
            logger.info("Intermediate Text is : {}".format(tmp_text.text))

            proposal_prod, proposal_token = self.get_MLM_prod(tmp_text, position, top_p, top_k)
            if proposal_token in self.MLM_tokenizer.all_special_ids: # 如果预测的token是special token，直接返回input_text
                logger.info("The proposal token is a special token, so return the original text.")
                return copy.deepcopy(input_text)

            proposal_token_prod = proposal_prod[proposal_token]
            proposal_tokenized_text = copy.deepcopy(tmp_text.tokenized_text) # token_type_ids和attention_mask应该是不变的
            # 第一个index为0，因为batch_size为1
            proposal_tokenized_text.input_ids[0][position] = proposal_token
            proposal_text = self.MLM_tokenizer.batch_decode(proposal_tokenized_text.input_ids, skip_special_tokens = True)[0]
            proposal_text_prob = self.get_whole_text_prod(proposal_text)

            A_insert = min(1, self.delete_prior * proposal_text_prob / (self.insert_prior * old_text_prob * proposal_token_prod))
            
            logger.info("Proposal Text is : {}".format(proposal_text))
            logger.info("Accept rate: {ar} ; Proposal text prob: {pp} ; Old text prob: {op} ; Proposal token prob: {ptp} ".format(ar=A_insert, pp=proposal_text_prob, op=old_text_prob, ptp=proposal_token_prod))
            if decision(A_insert):
                accept_text_hard_constraint_mask = copy.deepcopy(tmp_text.hard_constraint_mask)
                accept_text = ConstrainedGenText(proposal_text, proposal_tokenized_text, accept_text_hard_constraint_mask)
                return accept_text
            else:
                return copy.deepcopy(input_text)
            

if __name__ == "__main__":