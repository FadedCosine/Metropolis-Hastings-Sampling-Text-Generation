from datasets import load_dataset
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel, GPT2Tokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
from mhsg import MHSG, ConstrainedGenText
from nltk.stem.lancaster import LancasterStemmer
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
    parser.add_argument('--sample_time', type=int, default=300)
    parser.add_argument('--LM_model_file', type=str, default="model/gpt-2-1-billion")
    parser.add_argument('--MLM_model_file', type=str, default="/home/yangzhixian/pretrained_model/bert-large-uncased")
    parser.add_argument('--sent_data_path', type=str, default="data/1-billion/test_sent.txt")
    parser.add_argument('--key_data_path', type=str, default="data/1-billion/test_keywords.txt")
    parser.add_argument('--sample_num', type=int, default=500)
    parser.add_argument('--top_k',  type=int, default=0)
    parser.add_argument('--top_p',  type=float, default=0.6)
    args = parser.parse_args()
    logger.info(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with open(args.sent_data_path, 'r') as f:
        sents_list = f.readlines()[:args.sample_num]
    with open(args.key_data_path, 'r') as f:
        keywords_list = f.readlines()[:args.sample_num]
    LM_model = GPT2LMHeadModel.from_pretrained(args.LM_model_file)
    LM_tokenizer = GPT2Tokenizer.from_pretrained(args.LM_model_file)

    MLM_model = BertForMaskedLM.from_pretrained(args.MLM_model_file)
    MLM_tokenizer = BertTokenizer.from_pretrained(args.MLM_model_file)

    gen_model = MHSG(MLM_model, MLM_tokenizer, LM_model, LM_tokenizer)

  
    target_file = open("1_billion_target.txt", 'w')
    gen_file = open("1_billion_gen.txt", 'w')
    key_file = open("1_billion_key.txt", 'w')
    for target, keywords in zip(sents_list, keywords_list):
        key_concepts_list = keywords.strip().split()
        logger.info("Target is : {}".format(target))
        logger.info("Key is : {}".format(key_concepts_list))
        key_concepts_str = " ".join(key_concepts_list)
        key_concepts_tokenized = gen_model.MLM_tokenizer(key_concepts_str, return_tensors="pt")
        key_hard_constraint_mask = [0] * len(key_concepts_tokenized.input_ids[0])
        input_text = ConstrainedGenText(key_concepts_str, key_concepts_tokenized, key_hard_constraint_mask)
        position = -1
      
        target_file.write(target)
        key_file.write(key_concepts_str + '\n')
        for step in range(args.sample_time):
            position, action = gen_model.get_position_and_action(input_text, position)
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