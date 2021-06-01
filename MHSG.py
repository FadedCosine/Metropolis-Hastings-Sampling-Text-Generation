
class ConstrainedGenText:
    def __init__(self, tokenized_text, hard_constraint_mask):
        """
        tokenized_text 是经过PLM tokenize之后的 id list
        hard_constraint_mask 把hard_constraint对应位置设为0，其他位置设为1
        """
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


class MHSG:
    def __init__(self, MLM_model, LM_model):
        """
        MLM_model 是 transformers 库中的进行 Masked Language Modeling 的模型对象
        LM_model 是 transformers 库中的进行 Language Modeling的模型对象
        """
        self.MLM_model = MLM_model
        self.LM_model = LM_model
    
    