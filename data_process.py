from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.tag.mapping import tagset_mapping
from tqdm import tqdm
import random

random.seed(910)
PTB_UNIVERSAL_MAP = tagset_mapping('en-ptb', 'universal')
KEYWORD_POS = ["NN", "NNS", "NNP", "NNPS","JJ", "VB", "VBD"]
def to_universal(tagged_words):
    return [(word, PTB_UNIVERSAL_MAP[tag]) for word, tag in tagged_words]

def create_keywords_gen(read_filename, write_sent_filename, write_keywords_filename, min_seq_len=10, max_seq_len=20,sampler_max_num=5):

    write_sent_file = open(write_sent_filename, 'w')
    write_keywords_file = open(write_keywords_filename, 'w')
    with open(read_filename, 'r') as f:
        cnt = 0
        # print(len(list(f.readlines())))
        for line in tqdm(f.readlines(), mininterval=10,):
            line_split = word_tokenize(line)
            if len(line_split) < min_seq_len or len(line_split) > max_seq_len:
                continue
            pos_tagged = [(word, tag) for word, tag in pos_tag(line_split)]
            # universal_pos_tagged = to_universal(pos_tagged)
            key_num = random.randint(1, sampler_max_num)
            total_useful_tokens = []
            for item in pos_tagged:
                if item[1] in KEYWORD_POS:
                    total_useful_tokens.append(item[0])
            try:
                indices = random.sample(range(len(total_useful_tokens)), key_num)
                keywords_list = [total_useful_tokens[i] for i in sorted(indices)]
                write_keywords_file.write(" ".join(keywords_list) + '\n')
                write_sent_file.write(line)
            except:
                continue
    write_keywords_file.close()
    write_sent_file.close()

read_filename = "data/1-billion/heldout-monolingual.tokenized.shuffled/news.en-00000-of-00100"
write_sent_filename = "data/1-billion/test_sent.txt"
write_keywords_filename = "data/1-billion/test_keywords.txt"

create_keywords_gen(read_filename, write_sent_filename, write_keywords_filename)