from itertools import compress
import numpy as np
import random
import torch
import torch.nn.functional as F
class ConstrainedGenText:
    """
    Constraint就是一开始的text的构成
    """
    def __init__(self, text, mlm_tokenizer):
        """
        text 是原text，且所有字母小写
        tokenized_text 是经过PLM tokenize之后的 id list
        hard_constraint 是一个sequence的list
        """
        self.text = text.lower()
        self.mlm_tokenizer = mlm_tokenizer
        # tokenized_text与进行操作的text一定要一样
        self.tokenized_text = self.mlm_tokenizer(original_text, add_special_tokens=False, return_tensors="pt")
        self.hard_constraint_mask = torch.zeros(self.tokenized_text.input_ids.size())


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
    def get_position(self, input_text:ConstrainedGenText, last_position:int):
        """
        得到当前操作进行的位置
        Args:
            input_text (ConstrainedGenText): 输入的text
            last_position (int): 上一次进行操作的位置
        """
        aviable_indices = list(compress(range(len(test_list)), test_list))
        cur_position = random.choice(aviable_indices)
        assert cur_position < len(input_text.tokenized_text)
        return cur_position
    def replace(self, input_text:ConstrainedGenText, position:int, topp=0, topk=0):
        """
        进行替换操作，
        Args:
            input_text (ConstrainedGenText): [description]
            position (int): [description]
        """
        self.LM_model.eval()
        self.MLM_model.eval()
        with torch.no_grad():
            old_input_ids = self.LM_tokenizer(input_text.text, return_tensors='pt').input_ids
            old_target_ids = old_input_ids.clone()
            old_text_ppl = torch.exp(self.LM_model(old_input_ids, labels=old_target_ids)[0]).item()
            old_text_prob = old_text_ppl ** (-old_input_ids.size(0))

            logits = self.MLM_model(**input_text.tokenized_text).logits[0][position]

            if topp != 0 or topk != 0: # sampling
                logits = top_k_top_p_filtering(logits, top_k = top_k, top_p = top_p)
                prev_pred = F.softmax(logits, dim=-1)
                prev_token = prev_pred.multinomial(num_samples=1)
            else: # greedy search
                prev_pred = F.softmax(logits, dim=-1)
                prev_token = prev_pred.argmax(-1)
            prev_token_pred = prev_pred[prev_token]
            
