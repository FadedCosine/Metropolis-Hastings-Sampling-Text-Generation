import jsonlines
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from nltk.stem.lancaster import LancasterStemmer
from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn as nn
import  tqdm
import  argparse
import os
import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
lst = LancasterStemmer()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = nn.CrossEntropyLoss()
tokenizer = BertTokenizer.from_pretrained("~/pretrained_model/bert-base-uncased")
cache_data = "./commongen_data/cache_data"
class mytrainset(Dataset):
    def __init__(self):
        self.res = []
        with jsonlines.open('./commongen_data/commongen.train.jsonl') as reader:
            for line in reader:
                concept  = line['concept_set'].split("#")
                scene = line['scene']
                for sentence in scene:
                    sentence = sentence.split()
                    for cpt in concept:
                        for i in range(len(sentence)):
                            if lst.stem(sentence[i]) == cpt:
                                if len(tokenizer(sentence[i])["input_ids"]) >= 4: #tokenized成多个的sentence就不要了
                                    continue
                                masked_sent = sentence.copy()
                                masked_sent[i] = "[MASK]"
                                self.res.append([" ".join(masked_sent)," ".join(sentence)])
    def __getitem__(self, item):
        return self.res[item]
    def __len__(self):
        return len(self.res)



def train(args):
    if os.path.exists(cache_data):
        with open(cache_data, 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = mytrainset()
        with open(cache_data, 'wb') as f:
            pickle.dump(dataset, f)

    trainsampler = RandomSampler(dataset)
    trainloader = DataLoader(dataset, sampler=trainsampler, batch_size=args.btsize)
    model =BertForMaskedLM.from_pretrained("~/pretrained_model/bert-base-uncased")
    model = model.to(device)
    model = nn.DataParallel(model)
    model.train()
    optimzer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.n_epoch):
        for data in tqdm.tqdm(trainloader):
            # print(data[0])
            # print(data[1])
            inputs = tokenizer(data[0],return_tensors='pt',padding=True).to(device)
            # print("data[0] is : ", data[0])
            # print("inputs is : ", inputs)
            labels = tokenizer(data[1],return_tensors='pt',padding=True)["input_ids"].to(device)
            # print(inputs["input_ids"])
            # print(labels)
            # print(inputs["input_ids"].shape)
            # print(labels.shape)
            outputs =model(**inputs, labels=labels)
            loss = torch.mean(outputs.loss)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
    torch.save(model.state_dict(),"model/bert-base-uncased/pytorch_model.bin")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--btsize",type=int, default=128)
    parser.add_argument("--n_epoch", type=int, default=3)
    args = parser.parse_args()
    train(args)

main()