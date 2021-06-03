from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
from mhsg import MHSG, ConstrainedGenText
import random
import numpy as np
import torch
import argparse
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Metropolis Hastings Sampling Text Generation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_time', type=int, default=500)
    parser.add_argument('--LM_model_file', type=str, default="/home/yangzhixian/pretrained_model/gpt2_model")
    parser.add_argument('--MLM_model_file', type=str, default="/home/yangzhixian/pretrained_model/bert-base-uncased")
    parser.add_argument('--top_k',  type=int, default=0)
    parser.add_argument('--top_p',  type=float, default=0.9)
    args = parser.parse_args()
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = load_dataset('common_gen')

    LM_model = GPT2LMHeadModel.from_pretrained(args.LM_model_file)
    LM_tokenizer = GPT2Tokenizer.from_pretrained(args.LM_model_file)

    MLM_model = BertForMaskedLM.from_pretrained(args.MLM_model_file)
    MLM_tokenizer = BertTokenizer.from_pretrained(args.MLM_model_file)

    gen_model = MHSG(MLM_model, MLM_tokenizer, LM_model, LM_tokenizer)

    validation_num = len(dataset.data["validation"]["concepts"])
    target_file = open("validation_target.txt", 'w')
    gen_file = open("validation_gen.txt", 'w')
    key_file = open("validation_key.txt", 'w')
    for i in range(validation_num):
        key_concepts_list = [item.as_py() for item in dataset.data["validation"]["concepts"][i]]
  
        target = dataset.data["validation"]["target"][i]
        key_concepts_str = " ".join(key_concepts_list)
        key_concepts_tokenized = gen_model.MLM_tokenizer(key_concepts_str, return_tensors="pt")
        key_hard_constraint_mask = [0] * len(key_concepts_tokenized.input_ids[0])
        input_text = ConstrainedGenText(key_concepts_str, key_concepts_tokenized, key_hard_constraint_mask)
        position = -1
        logger.info("Original Text is : {}".format(key_concepts_str))
        logger.info("Target is : {}".format(target))
        target_file.write(target.as_py() + '\n')
        key_file.write(key_concepts_str + '\n')
        for step in range(args.sample_time):
            position, action = gen_model.get_position_and_action(input_text, position)
            if step < 10: # 前十步强制插入
                action = "insert"
            if action == "replace": # 替代 
                input_text = gen_model.replace(input_text, position, args.top_p, args.top_k)
            elif action == "insert": #插入
                input_text = gen_model.insert(input_text, position, args.top_p, args.top_k)
            elif action == "delete": #删除
                input_text = gen_model.delete(input_text, position, args.top_p, args.top_k)
            logger.info("Sampling step {}, Text is : {}".format(step, input_text.text))
        gen_file.write(input_text.text + '\n')
    target_file.close()
    gen_file.close()
    key_file.close()
if __name__ == "__main__":
    main()