import sys
import os
from pathlib import Path
from unittest.util import _MAX_LENGTH

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())

sys.path.append(CURR_DIR)
P = PATH.parent
print("current dir: ",CURR_DIR)
for i in range(1):  # add parent path, height = 3
    P = P.parent
    PROJECT_PATH = str(P.absolute())
    sys.path.append(str(P.absolute()))


from torch.utils.data import Dataset
import torch.utils.data as util_data
from tqdm import tqdm
import torch
from collections import Counter
from utils.randomness import set_global_random_seed
import random
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import transformers
from torch.utils.data import Dataset
import json
from utils.pickle_picky import *
from data_converter import *

def multi_acc(y_pred, y_true):
  acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_true).sum().float() / float(y_true.size(0))
  return acc


class mnli_data(Dataset):
    def __init__(self, texts, labels) -> object:
        self.texts = texts
        self.labels = labels
        print("各类数量:",Counter(self.labels))
        assert len(self.texts)==len(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            'texts': self.texts[idx],
            'labels': self.labels[idx]}

def fine_tune_v3(args):

    with open(args.config_path) as file:
        config = json.load(file)
    
    set_global_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.cuda_index))
    args.device = device
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'],_MAX_LENGTH=32)
    model = AutoModelForSequenceClassification.from_pretrained(config['model_path'], num_labels=3)
    model = model.to(device)

    # 加载数据 
    sortByPlabelConfidence = pickle_load(args.sortByPlabelConfidence_path)
    train_data = pickle_load(args.train_path)
    label2id = pickle_load(config['label2id'])
    templateMapping = config['template_mapping']
    label2template = {}
    for k, v in templateMapping.items():
        label2template[label2id[k]] = v
    data = prepare4mnil_dynamic(sortByPlabelConfidence, train_data,  args.ratio, label2template, args.dynamic_num)

    texts = [f"{item.premise} {tokenizer.sep_token} {item.hypothesis}"  for item in data]
    labels = [item.label for item in data]
    
    print("\tdata convert complete")

    args.data_num = len(data)//3
    random.shuffle(data) # 打乱
    texts = [f"{item.premise} {tokenizer.sep_token} {item.hypothesis}."  for item in data]
    labels = [item.label for item in data]
    n_dev = int(len(data)*0.2) 
    # 划分dev, train, 创建data_loader
    dev_dataset = mnli_data(texts[:n_dev],labels[:n_dev])
    train_dataset = mnli_data(texts[n_dev:],labels[n_dev:])
    dev_loader = util_data.DataLoader(dev_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    train_loader = util_data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)


    # create optimizer
    optimizer = AdamW(model.parameters(), lr=config['lr'], correct_bias=False) #  一般设定 4e-7 (succeeded experiment)   /2e-6 gt


    warm_up_steps = int(args.ratio*100*40)
    print("warm up steps:",warm_up_steps)
    scheduler = transformers.get_cosine_schedule_with_warmup(                                    
        optimizer,
        num_warmup_steps=warm_up_steps,
        num_training_steps = config['epoch']*(len(train_loader))
        )
    

    N_print = 1
    # train
    model.train()
    base_acc = -1
    base_train_loss = 1e9
    for epoch in range(config['epoch']):
        total_step = len(train_loader)
        accs = []
        losses = []
        for batch_idx, batch in enumerate(train_loader):
            text = batch['texts']
            label = batch['labels']
            input_ids = tokenizer.batch_encode_plus(text, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(device)
            out = model(input_ids,labels=label.to(device))
            loss = out[0]
            prediction = out[1]
            loss.backward()
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


            # 记录
            acc = multi_acc(prediction.detach().cpu(), label)
            accs.append(acc.item())
            losses.append(loss.detach().cpu().item())
            if batch_idx%N_print==0:
                print('\r', " step {}/{} ,  loss_{:.4f} acc_{:.4f}  lr:{} \r".format(batch_idx+1,total_step,np.mean(losses),np.mean(accs),lr), end='', flush=True)
        
        train_acc  = np.mean(accs)
        train_loss = np.mean(losses)
        if train_loss>base_train_loss:
            # 如果当前epoch的train loss > 前一个epoch的train loss, 退出
            # break
            pass
        else:
            base_train_loss = train_loss


        # evaluation
        with torch.no_grad():
            dev_accs = []
            for batch_idx, batch in enumerate(dev_loader):
                text = batch['texts']
                label = batch['labels']
                input_ids = tokenizer.batch_encode_plus(text, padding=True, truncation=True)
                input_ids = torch.tensor(input_ids["input_ids"]).to(device)
                prediction = model(input_ids,labels=label.to(device))[-1]
                acc = multi_acc(prediction.detach().cpu(), label)
                dev_accs.append(acc.detach().cpu().item())
            dev_acc = np.mean(dev_accs)

        # 保存模型
        if dev_acc>base_acc:
            base_acc = dev_acc
            print("save model to ",args.check_point_path)
            print("dev acc is:{}".format(dev_acc))
            torch.save(model.state_dict(),args.check_point_path)
        print()
        print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}  dev_acc: {dev_acc:.4f}')







    






if __name__=="__main__":
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser()


    parser.add_argument('--check_point_path',type=str,default="",
                        help = "finetune好的模型保存到哪里")
    parser.add_argument('--ratio', type=float,default=0.09, help='用0.05*Ntrain的数据finetune')
    parser.add_argument("--config_path", type=str,default="/data/tywang/transfer/tac_configs/config_tac_partial_constrain.json", help="as named")

    parser.add_argument("--sortByPlabelConfidence_path", type=str,default='/data/tywang/transfer/tac_configs/NLNL_tac_RT0.05_SD17.pkl', 
                        help="select数据的路径")
    parser.add_argument('--train_path',type=str,default="",
                        help = "as named")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--cuda_index', type=int)

    parser.add_argument('--dynamic_num', type=int)
    


    args = parser.parse_args()
    fine_tune_v3(args)


    debug_stop = 1  


