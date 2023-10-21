
import sys
import os
from pathlib import Path

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
import time
TIME=time.strftime("%m%d%H%M%S", time.localtime())# 记录被初始化的时间
print("time",TIME)
from utils.dict_relate import dict_index
from utils.randomness import set_global_random_seed
from model.sccl import SCCL_BERT
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, dataset
import copy
import torch.nn.functional as F
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse
import torch.utils.data as util_data
import numpy as np
from tqdm import tqdm
from collections import Counter
import json
from utils.pickle_picky import *
from dataclasses import dataclass

class Train_dataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __getitem__(self, index):
        return {
            'text': self.data['initText'][index],
            'p_label': self.data['p_label'][index],  # pseudo_label
            'index': self.data['index'][index],
            'label': self.data['label'][index]
        }

    def __len__(self):
        return len(self.data['text'])

def lable_transfer(init_label: list, label2id):
    return [label2id[label] for label in init_label]

def lableid_transfer(init_label: list, label2id, init_id2label):
    return [label2id[init_id2label[label]] for label in init_label]

@dataclass
class PlabelWithConfidenceItem:
    dataIndex: int
    labelIndex: int
    confidence: float
    predictLabelIndex: InterruptedError

def sortByPlabelConfidence(train_preds_hist, train_dataset, mode, num_classes=35):
    plabelWithConfidence=[]
    allPlabel=[]
    weightV2 = torch.FloatTensor(num_classes).zero_() + 1.
    for index, (trainData, preds) in enumerate(zip(train_dataset, train_preds_hist.mean(1))):
        allPlabel.append(trainData['p_label'])
        plabelWithConfidence.append(
            PlabelWithConfidenceItem(
                index, 
                trainData['p_label'], 
                preds[trainData['p_label']],
                np.argmax(preds.numpy())
            )
        )
    sortedPlabelConfidence = sorted(plabelWithConfidence, key= lambda item : item.confidence, reverse=True)

    ratios = [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.1, 0.5,1]
    select_num = [int(rt*len(plabelWithConfidence)) for rt in ratios]
    rightClass=[]
    top_09 = 0
    top_09_class = 0
    for select_n,rt in zip(select_num, ratios):
        selectPlabelData  = sortedPlabelConfidence[:int(select_n)]
        cnt_right, cnt_wrong = 0, 0
        for item in selectPlabelData:
            if rt == 0.1:
                weightV2[item.labelIndex] +=1 
            if item.labelIndex == train_dataset[item.dataIndex]['label']:
                cnt_right+=1
                rightClass.append(item.labelIndex)
            else:
                cnt_wrong+=1
        acc = cnt_right / (cnt_right + cnt_wrong + 0.00005)
        print("(only positive) Top {} {} confident Data acc= {}, class num:{}, class: {}".format(rt, select_n, acc,
                                                            len(set(rightClass)), Counter(rightClass)))
        if rt == 0.5:
            top_09 = acc
            top_09_class = len(set(rightClass))
    return sortedPlabelConfidence, top_09, top_09_class, weightV2


def NLNL_main(args):
    with open(args.config_path) as file:
        config = json.load(file)
    set_global_random_seed(config['seed'])
    label2id = pickle_load(config['label2id_path'])
    id2label = {v:k for k,v in label2id.items()}
    init_label2id = pickle_load(config['init_label2id_path'])
    init_id2label = {v:k for k,v in init_label2id.items()}


    num_classes = config['n_rel']
    train_data = pickle_load(args.train_path)
    train_data['top1'] = lableid_transfer(train_data['top1'], label2id, init_id2label)
    train_data['label'] = lableid_transfer(train_data['label'], label2id, init_id2label)
    train_data['p_label'] = train_data['top1']
    N_train = len(train_data['text'])
    train_data['index'] = [i for i in range(N_train)]

    dev_data = pickle_load(args.dev_path)
    dev_data['top1'] = lableid_transfer(dev_data['top1'], label2id, init_id2label)
    dev_data['label'] = lableid_transfer(dev_data['label'], label2id, init_id2label)
    dev_data['p_label'] = dev_data['top1']
    N_dev = len(dev_data['text'])
    dev_data['index'] = [i for i in range(N_dev)]

    print("*"*10,"information","*"*10)
    info_acc = sum(np.array(train_data['p_label'])==np.array(train_data['label']))/len(train_data['p_label'])
    n_pseudo_label_relation = len(set(train_data['p_label']))
    print("acc:{:.4f}".format(info_acc))
    print("n_relation:{}".format(n_pseudo_label_relation))
    print("N_data:{}".format(len(train_data['p_label'])))
    print("*"*10,"***********","*"*10)
    


    train_dataset = Train_dataset(train_data)
    train_loader = util_data.DataLoader(train_dataset, batch_size=config['batch'], shuffle=True, num_workers=0)
    dev_dataset = Train_dataset(dev_data)
    dev_loader = util_data.DataLoader(dev_dataset, batch_size=config['batch'], shuffle=True, num_workers=0)

    tags = pickle_load(args.e_tags_path)
    device = torch.device('cuda:{}'.format(config['cuda_index']))
    bert_model = SentenceTransformer(config['model_dir'])
    sccl_model:SCCL_BERT = SCCL_BERT(bert_model, config['max_len'],
                           device, config['n_rel'], True, tags).to(device)

    ##
    optimizer = torch.optim.AdamW([
        {'params': sccl_model.sentbert.parameters()},
        {'params': sccl_model.out.parameters(), 'lr': config['lr']*config['lr_scale']}], lr=config['lr'])
    ##

    # 产生每个类的weight
    weight = torch.FloatTensor(num_classes).zero_() + 1.
    for i in range(num_classes):
        weight[i] = (torch.from_numpy(np.array(train_data['p_label']).astype(int)) == i).sum()  
    weight = 1 / (weight / weight.max()).to(device)
    weight = torch.where(torch.isinf(weight), torch.full_like(weight, 1), weight)
    ##

    criterion = nn.CrossEntropyLoss()
    criterion_nll = nn.NLLLoss()
    criterion_nr = nn.CrossEntropyLoss(reduction='none')  # compute per-sample losses
    criterion.to(device)
    criterion_nll.to(device)
    criterion_nr.to(device)
    ##

    ##
    # NLNL parameters
    num_hist = 10
    train_preds = torch.zeros(N_train, num_classes) - 1.
    train_preds_hist = torch.zeros(N_train, num_hist, num_classes)
    dev_preds = torch.zeros(N_dev, num_classes) - 1.
    dev_preds_hist = torch.zeros(N_dev, num_hist, num_classes)

    pl_ratio = 0.
    nl_ratio = 1.-pl_ratio  
    train_losses = torch.zeros(N_train) - 1.  # 每个数据的loss
    ##

    ##
    # train
    best_top09 = -np.inf
    best_topClass = 0
    for epoch in range(config['epoch']):
        train_loss = train_loss_neg = train_acc = 0.0
        pl = 0.; nl = 0.
        sccl_model.train()
        accs = []
        losses = []
        data_predict = torch.zeros(N_train,dtype=torch.long) # 预测值
        data_predict_confidence = torch.zeros(N_train) # 预测值的confidence
        for i, data in enumerate(train_loader):
            
            text, labels, index = data['text'],data['p_label'],data['index']
            labels_neg = (labels.unsqueeze(-1).repeat(1, config['ln_neg'])
                      + torch.LongTensor(len(labels), config['ln_neg']).random_(1, num_classes)) % num_classes
            assert labels_neg.max() <= num_classes-1
            assert labels_neg.min() >= 0
            assert (labels_neg != labels.unsqueeze(-1).repeat(1, config['ln_neg'])
                    ).sum() == len(labels)*config['ln_neg']  # 保证得到的都是和原来label不同的数据
            labels = labels.to(device)
            labels_neg = labels_neg.to(device)
            logits = sccl_model.out(sccl_model.get_embeddings_PURE(text))

            s_neg = torch.log(torch.clamp(1.-F.softmax(logits, -1), min=1e-5, max=1.))  # log(1-pk)
            s_neg *= weight[labels].unsqueeze(-1).expand(s_neg.size()).to(device)
            _, pred = torch.max(logits.data, -1)  

            ##
            # find labels for fewshot
            confidences = F.softmax(logits,-1)
            confidence = confidences[np.array([i for i in range(len(logits))]),pred]
            data_predict[index] = pred.detach().cpu() # 记录每个train data的prediction
            data_predict_confidence[index] = confidence.detach().cpu() # 记录每个train数据的prediction的confidence
            stop = 1
            ##

            acc = float((pred == labels.data).sum())   # batch的正确个数
            train_acc += acc
            accs.append(acc/len(index))
            
            loss_neg = criterion_nll(s_neg.repeat(config['ln_neg'], 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
            
            loss = (loss_neg) / (float((labels >= 0).sum()) +float((labels_neg[:, 0] >= 0).sum()))
            loss.backward()
            optimizer.step()
            l = logits.size(0)*loss.detach().cpu().data
            train_loss+=l
            
            losses.append(l/logits.size(0))
            train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data

            print('\r', "EPOCH[{}] step {}/{} ,  loss: {:.4f} train_acc: {:.4f}  ".format(epoch+1,i+1,len(train_loader),
                                                                                          l/logits.size(0),np.mean(accs)), end='', flush=True)


        train_preds_hist[:, epoch % num_hist] = train_preds
        sortedPlabelConfidence, _, _, weightV2 = sortByPlabelConfidence(train_preds_hist, train_dataset, 0, config['n_rel'])
        
        #[dynamic weight]
        weightV2 = (weightV2 / weightV2.max()).to(device)
        weightV2 = torch.exp(0.5 - weightV2/torch.max(weightV2))
        print(weightV2)
        weight *= weightV2
        print(weight)

        # [dev dataset]
        sccl_model.eval()
        for i, data in enumerate(dev_loader):
            text, labels, index = data['text'],data['p_label'],data['index']
            logits = sccl_model.out(sccl_model.get_embeddings_PURE(text))
            dev_preds[index.cpu()] = F.softmax(logits, -1).cpu().data
        dev_preds_hist[:, epoch % num_hist] = dev_preds
        _, top09, top09_class, _ = sortByPlabelConfidence(dev_preds_hist, dev_dataset, 1, config['n_rel'])

        # if top09_class > best_topClass:
        #     print("[NEW BEST]: best_09: {}, best_09_class: {}".format(top09, top09_class))
        #     best_top09 = top09
        #     best_topClass = top09_class
        #     pickle_save(sortedPlabelConfidence, os.path.join(args.specified_save_path, 
        #                                                      "sortedPlabelConfidence_{}.pkl".format(TIME)))
        sort_index = []
        for item in sortedPlabelConfidence:
            item.labelIndex = init_label2id[id2label[item.labelIndex]]
            item.predictLabelIndex = init_label2id[id2label[item.predictLabelIndex]]
            sort_index.append(item.dataIndex)
        
        if top09_class > best_topClass:
            print("[NEW BEST]: best_09: {}, best_09_class: {}".format(top09, top09_class))
            best_top09 = top09
            best_topClass = top09_class
            pickle_save(sort_index, args.sortedPlabelConfidence_path)
        elif top09 > best_top09 and top09_class == best_topClass:
            print("[NEW BEST]: best_09: {}, best_09_class: {}".format(top09, top09_class))
            best_top09 = top09
            pickle_save(sort_index, args.sortedPlabelConfidence_path)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sortedPlabelConfidence_path", type=str,default="/home/jwwang/URE_share/outcome/NL/clean_data", help="as named")
    
    parser.add_argument("--train_path", type=str,
                        default="/data/tywang/URE_share/data/tac/tac_pseudo_pos.pkl", help="as named")
    parser.add_argument("--dev_path", type=str,
                        default="/data/tywang/URE_share/data/tac/tac_pseudo_pos.pkl", help="as named")
    parser.add_argument("--e_tags_path", type=str,
                        default="/data/tywang/URE_share/data/tac/tags.pkl", help="as named")
    parser.add_argument("--config_path", type=str,
                        default="/home/jwwang/URE_UEE/NL/config/ace.json", help="as named")

    args = parser.parse_args()

    NLNL_main(args)

    