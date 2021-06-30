from    tqdm import tqdm
from    transformers import GPT2LMHeadModel, GPT2TokenizerFast
import  argparse
import  os
import torch
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S')
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


@torch.no_grad()
def cal_ppl(filenames, model_file):
    ppl_dir = {}
    nnl_dir = {}
    for filename in filenames:
        with open(filename, 'r') as f:
            sents = [line.strip('\n').strip().lower() for line in f.readlines()]
        model = GPT2LMHeadModel.from_pretrained(model_file).cuda()
        tokenizer = GPT2TokenizerFast.from_pretrained(model_file)
        model.eval()
        ppl = torch.FloatTensor(len(sents)).cuda()
        nnl = torch.FloatTensor(len(sents)).cuda()
        max_length = model.config.n_positions
        for index, sent in tqdm(enumerate(sents)):
            encodings = tokenizer(sent, return_tensors='pt')
            input_ids = encodings.input_ids[:, :max_length].cuda()
            target_ids = input_ids.clone()
            outputs = model(input_ids, labels=target_ids)
            print(outputs)
            nnl[index] = outputs[0].tolist()
            ppl[index] = torch.exp(outputs[0]).tolist()
        ppl_dir[filename] = torch.mean(ppl)
        nnl_dir[filename] = torch.mean(nnl)
    for i in ppl_dir.keys():
        logger.info('PPL : {:<65}{:.3f}'.format(os.path.basename(i), ppl_dir[i]))
        logger.info('NNL : {:<65}{:.3f}'.format(os.path.basename(i), nnl_dir[i]))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-p',type=float, default=0.9)
    parser.add_argument('--top-k',type=int, default=0)
    parser.add_argument('--model_file', type=str)
    parser.add_argument('--cuda_num', type=str, default='7')
    parser.add_argument('--file_path', type=str, default="1_billion_report.txt")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num

    filenames = [args.file_path]
    logger.info(filenames)
    cal_ppl(filenames, args.model_file)