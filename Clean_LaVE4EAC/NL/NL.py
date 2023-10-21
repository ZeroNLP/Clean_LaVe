
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
from utils.pickle_picky import pickle_load, pickle_save
from utils.randomness import set_global_random_seed
from model.baseBert import EAE_BERT
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, dataset
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
from dataclasses import dataclass
import json
from collections import defaultdict
import random
from torch.nn.utils.rnn import pad_sequence
import collections
import queue
import time
import logging
from utils.printUtils import *



class Train_dataset(Dataset):
    def __init__(self, dataset, label2id):
        self.data = dataset
        self.label2id=label2id

    def __getitem__(self, index):
        return {
            'text': self.data['text'][index],
            'text_piece': self.data['text_piece'][index],
            'p_label': self.data['p_label'][index],
            'label': self.label2id[self.data['role'][index]],
            'trigger': self.data['trigger'][index], 
            'trigger_be': self.data['trigger_be'][index],
            'argument': self.data['argument'][index],
            'argument_be': self.data['argument_be'][index],
            'index': index
        }

    def __len__(self):
        return len(self.data['text'])
    
    @staticmethod
    def collate_fn(data):
        collData = defaultdict(list)
        for item in data:
            for k, v in item.items():
                collData[k].append(v)
        return collData


@dataclass
class PlabelWithConfidenceItem:
    dataIndex: int
    labelIndex: int
    confidence: float
    predictLabelIndex: InterruptedError

@dataclass
class LogitItem:
    dataIndex: int

    labelIndex: int
    confidence: float

    gtLabelIndex: list


printMapping = [redPrint, greeenPrint]




def sortByLogitsInfo(train_preds, train_dataset, mode):
    logitsData=[]
    cnt_right , cnt_all = 0, 0
    cnt_class =[]
    pick_dataIndex = []
    pick_0 = 0
    pick_rt = 0.05
    for index, logit in enumerate(train_preds):
        logitsData.append(
            LogitItem(
                index, 
                np.argmax(logit.numpy(), -1),
                np.max(F.softmax(logit, -1).numpy(), -1),
                train_dataset[index]['label']
            )
        )
        cnt_all +=1
        if np.argmax(logit.numpy(), -1) == train_dataset[index]['label']:
            cnt_right +=1
            cnt_class.append(np.argmax(logit.numpy(), -1))
    
    printMapping[mode]("According to new Confidence: acc:{}, diversity:{}".format(cnt_right/cnt_all, len(set(cnt_class))))
    ####### 判断新的logit正确的特点
    ratios = [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.5,1]
    sorted_logitsData = sorted(logitsData, key= lambda item: item.confidence, reverse= True)
    ratios = [0.01,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.075,0.08,0.09,0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.8, 0.9, 1]
    select_num = [int(rt*len(sorted_logitsData)) for rt in ratios]
    for select_n,rt in zip(select_num, ratios):
        selectData = sorted_logitsData[:int(select_n)]
        cnt_right, cnt_all =0, 0
        rightClass =[]
        for item in selectData:
            if rt == pick_rt:
                if item.labelIndex == 0:
                    pick_dataIndex.append(item.dataIndex)
                if item.gtLabelIndex == 0:
                    pick_0 += 1


            if item.labelIndex < 23:
                cnt_all +=1
                if item.labelIndex == item.gtLabelIndex:
                    cnt_right +=1
                    rightClass.append(item.labelIndex)
        printMapping[mode]("(only positive) ************ Top {} {} confident Data acc= {}, num of class:{}, right class: {}".format(rt, select_n, cnt_right/cnt_all,
                                                            len(set(rightClass)), Counter(rightClass)))
    
    return sorted_logitsData, pick_dataIndex, pick_0/int(pick_rt*len(sorted_logitsData))

def sortByPlabelConfidence(train_preds_hist, train_dataset, mode, num_classes=22):
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
            if item.labelIndex < 23:
                if rt == 0.1:
                    weightV2[item.labelIndex] +=1 
                if item.labelIndex == train_dataset[item.dataIndex]['label']:
                    cnt_right+=1
                    rightClass.append(item.labelIndex)
                else:
                    cnt_wrong+=1
        acc = cnt_right / (cnt_right + cnt_wrong + 0.00005)
        printMapping[mode]("(only positive) Top {} {} confident Data acc= {}, class num:{}, class: {}".format(rt, select_n, acc,
                                                            len(set(rightClass)), Counter(rightClass)))
        if rt == 0.09:
            top_09 = acc
            top_09_class = len(set(rightClass))
    return sortedPlabelConfidence, top_09, top_09_class, weightV2

def lableid_transfer(init_label: list, label2id):
    return [label2id[label] for label in init_label]

def pick_dict(data: dict, filter_index):
    pick_data = {}
    for k, v in data.items():
        pick_data[k] = [item for index, item in enumerate(v) if index not in filter_index]
    return pick_data

def filter_dataloader(data: list, filter_index: list, label2id, bsz: int):
    data_ = pick_dict(data, filter_index)
    train_dataset = Train_dataset(data_, label2id=label2id)
    train_loader = util_data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=0, 
                                        collate_fn=Train_dataset.collate_fn)
    return data_, train_dataset, train_loader

def NLNL_main(args):

    with open(args.config_path) as file:
        config = json.load(file)

    set_global_random_seed(args.seed)
    label2id = pickle_load(config['label2id_path'])
    id2label = {v:k for k,v in label2id.items()}
    init_label2id = pickle_load(config['init_label2id_path'])
    init_id2label = {v:k for k,v in init_label2id.items()}

    num_classes = config['n_rel']
    train_data = pickle_load(args.train_path)
    train_data['p_label'] = lableid_transfer(train_data['output'], label2id)
    train_data['output_id'] = lableid_transfer(train_data['output'], label2id)
    train_data['role_id'] = lableid_transfer(train_data['role'], label2id)
    train_data, train_dataset, train_loader = filter_dataloader(train_data, [], label2id, config['batch'])
    dev_data = pickle_load(args.dev_path)
    dev_data['p_label'] = lableid_transfer(dev_data['output'], label2id)
    dev_data['output_id'] = lableid_transfer(dev_data['output'], label2id)
    dev_data['role_id'] = lableid_transfer(dev_data['role'], label2id)
    dev_data, dev_dataset, dev_loader = filter_dataloader(dev_data, [], label2id, config['batch'])
    


    device = torch.device('cuda:{}'.format(args.cuda))
    bert_model = SentenceTransformer(config['model_dir'])
    sccl_model:EAE_BERT = EAE_BERT(bert_model, config['max_len'],
                           device, config['n_rel'], True).to(device)
    print("==> modeling loading complete")

    ##
    # set training related
    # net:NLNL_Net = NLNL_Net(sccl_model).to(device)
    optimizer = torch.optim.AdamW([
        {'params': sccl_model.sentbert.parameters()},
        {'params': sccl_model.out.parameters(), 'lr': config['lr']*config['lr_scale']},
        # {'params': sccl_model.factor, 'lr': 10000000}
        # {'params': sccl_model.dynamic_weight.parameters(), 'lr': config['lr']*config['lr_scale'] },
        # {'params': sccl_model.task_variant, 'lr': config['lr']**config['lr_scale']}
        ], lr=config['lr'])
    ##

    # 产生每个类的weight
    weight = torch.FloatTensor(num_classes).zero_() + 1.
    for i in range(num_classes):
        weight[i] = (torch.from_numpy(np.array(train_data['p_label']).astype(int)) == i).sum()  
    weight = 1/(weight / weight.max()).to(device)
    weight = torch.where(torch.isinf(weight), torch.full_like(weight, 1), weight)
    print(weight)
    # weight[-1] = torch.mean(weight[0:-1], dim= -1)
    # sccl_model.dynamic_weight[0].weight.data = weight
    # print(list(sccl_model.dynamic_weight.named_parameters()))
    # print(weight) 



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
    num_hist = 5
    N_train = len(train_data['text'])
    print(N_train)
    train_preds = torch.zeros(N_train, num_classes) - 1.
    train_preds_hist = torch.zeros(N_train, num_hist, num_classes)

    N_dev = len(dev_data['text'])
    print(N_dev)
    dev_preds = torch.zeros(N_dev, num_classes) - 1.
    dev_preds_hist = torch.zeros(N_dev, num_hist, num_classes)
    ##

    ##
    # candidate pool:
    candidate_index_pool= []
    candidate_index_pool_label = []
    ##

    ##
    # train
    recordAvgLoss = queue.Queue()
    best_top09 = -np.inf
    best_topClass = 0
    for epoch in range(config['epoch']):
        print(f"[EPOCH:{epoch}]")
        sccl_model.train()
        topAccs = []
        totalLosses = []
        for i, data in enumerate(train_loader):
            
            text, labels, index = data['text'],data['p_label'],data['index']
            # print(labels)
            # print(labels)
            # for label in labels:
            #     # print(label)
            #     print(len(np.setdiff1d(np.arange(0, num_classes), np.array(label))))
            labels_neg = torch.tensor([
                random.choices(list(np.setdiff1d(np.arange(0, num_classes), np.array([label]))), k=config['ln_neg'])
                for label in labels
            ])
            
            # labels_tensor = torch.zeros()
            # labels_
            # padding_labels = pad_sequence(labels_tensor, batch_first=True,  padding_value= 0)
            # print(labels_tensor)
            # print(padding_labels)
            # print("**************")

            # print(labels)
            # labels = torch.tensor(labels).to(device)
            labels_neg = labels_neg.to(device)
            logits = sccl_model.out(sccl_model.get_embeddings(text,
                                                              data['text_piece'],
                                                              data['trigger_be'],
                                                              data['argument_be']))

            # initial
            s_neg = torch.log(torch.clamp(1.-F.softmax(logits, -1), min=1e-5, max=1.))  # log(1-pk)
            
            # weight = sccl_model.dynamic_weight(labels.unse)
            #! 2
            s_neg *= weight

            # 3
            # s_neg *= sccl_model.get_weight(labels)
            # avgWeight = torch.tensor([ torch.mean(sccl_model.factor[label]) for label in labels]) #!tensor操作不可导
            
            # avgWeight = torch.tensor([torch.mean(weight[label]) for label in labels])
            # s_neg *= avgWeight.unsqueeze(-1).expand(s_neg.size()).to(device) #-- class-weight or not why not dynamic
            loss_neg = criterion_nll(s_neg.repeat(config['ln_neg'], 1), labels_neg.t().contiguous().view(-1))

            loss = sccl_model.combine_loss(loss_neg)
            loss.backward()
            # print(sccl_model.sentbert.parameters())
            # print(list(sccl_model.sentbert.parameters())[-6].grad)
            # print(list(sccl_model.sentbert.parameters())[0].requires_grad)
            optimizer.step()
            train_preds[index] = F.sigmoid(logits).cpu().data
            totalLosses.append(loss.detach().cpu().numpy())

            cnt_right, cnt_whole, cnt_pright = 0,0, 0
            for plabel, label, plabel_gt in zip(np.argmax(train_preds[index].numpy(), -1), data['label'], data['p_label']):
                if plabel < 23:
                    cnt_whole+=1
                    if plabel == label:
                        cnt_right+=1
                    if plabel == plabel_gt:
                        cnt_pright +=1 
            topAccs.append(cnt_right/(cnt_whole+0.005))

            # print('*' * 80)
            # print(loss)
            # print(plLoss)
            # print(loss_neg)
            # print('*' * 80)
            print('\r', "(only positive) EPOCH[{}] step {}/{}, top acc: {}, top pacc: {}, neg_loss: {},  total_loss: {}".format(epoch+1,i+1,
                                                                                               len(train_loader), 
                                                                                               cnt_right/(cnt_whole + 0.0005),
                                                                                               cnt_pright/(cnt_whole+ 0.0005),
                                                                                                loss_neg,
                                                                                                loss), end='', flush=True)
            
        
        # Early termination 
        # if recordAvgLoss.qsize() < 5:
        #     print("[Update]: lastAvgLoss:{}".format(np.average(totalLosses)))
        #     recordAvgLoss.put(np.average(totalLosses))
        # else:
        #     if epoch >= config['candidate_epoch'] and np.average(recordAvgLoss.queue) < np.average(totalLosses):
        #         print("[ATTENTION]: loss increse, break now!!!")
        #         print(recordAvgLoss.queue)
        #         print(np.average(totalLosses))
        #         break   
        #     else:
        #         print("[Update]: lastAvgLoss:{}->{}".format(recordAvgLoss.queue[-1], np.average(totalLosses)))
        #         recordAvgLoss.get()
        #         recordAvgLoss.put(np.average(totalLosses))
                
        
        print()
        print('\r', "===> EPOCH[{}] average top acc: {}".format(epoch+1, np.mean(topAccs)), end='\n', flush=True)


        ## 根据新的logits来选择数据:
        sorted_logitsData, train_index, train_ratio = sortByLogitsInfo(train_preds, train_dataset, 0)
        
        #---------------------------------------------------------------------------------
        ##  根据plabel的confidence来选择数据
        train_preds_hist[:, epoch % num_hist] = train_preds
        sortedPlabelConfidence, _, _, weightV2 = sortByPlabelConfidence(train_preds_hist, train_dataset, 0, config['n_rel'])


        # init_len = len(train_data['p_label'])
        # train_data, train_dataset, train_loader = filter_dataloader(train_data, train_index, label2id, config['batch'])
        # print("shrink dataset from {} to {}".format(init_len, len(train_data['p_label'])))

        N_train = len(train_data['text'])
        print(N_train)
        train_preds = torch.zeros(N_train, num_classes) - 1.
        train_preds_hist = torch.zeros(N_train, num_hist, num_classes)



        #! update weight 
        weightV2 = (weightV2 / weightV2.max()).to(device)
        weightV2 = torch.exp(0.5 - weightV2/torch.max(weightV2))
        print(weightV2)
        weight *= weightV2
        print(weight)


        # pickle_save(sortedPlabelConfidence, "/data/jwwang/URE_UEE/output/NL/sortedPlabelConfidence.pkl")


        sccl_model.eval()
        for i, data in enumerate(dev_loader):
            text, labels, index = data['text'],data['p_label'],data['index']
            logits = sccl_model.out(sccl_model.get_embeddings(text,
                                                              data['text_piece'],
                                                              data['trigger_be'],
                                                              data['argument_be']))
            dev_preds[index] = F.sigmoid(logits).cpu().data
        
        _, dev_index, dev_ratio = sortByLogitsInfo(dev_preds, dev_dataset, 1)
        dev_preds_hist[:, epoch % num_hist] = dev_preds
        _, top09, top09_class, _ = sortByPlabelConfidence(dev_preds_hist, dev_dataset, 1, config['n_rel'])


        # convert id
        for item in sortedPlabelConfidence:
            item.labelIndex = init_label2id[id2label[item.labelIndex]]
            item.predictLabelIndex = init_label2id[id2label[item.predictLabelIndex]]

        for item in sorted_logitsData:
            item.labelIndex = init_label2id[id2label[item.labelIndex]]
            item.gtLabelIndex = init_label2id[id2label[item.gtLabelIndex]]
        
        if top09_class > best_topClass:
            print("[NEW BEST]: best_09: {}, best_09_class: {}".format(top09, top09_class))
            best_top09 = top09
            best_topClass = top09_class
            pickle_save(sortedPlabelConfidence, args.sortedPlabelConfidence_path)
            # pickle_save(sorted_logitsData, "/data/jwwang/URE_UEE/output/NL/sortedLogitData_{}.pkl".format(TIME))
        elif top09 > best_top09 and top09_class == best_topClass:
            print("[NEW BEST]: best_09: {}, best_09_class: {}".format(top09, top09_class))
            best_top09 = top09
            pickle_save(sortedPlabelConfidence, args.sortedPlabelConfidence_path)
            # pickle_save(sorted_logitsData, "/data/jwwang/URE_UEE/output/NL/sortedLogitData_{}.pkl".format(TIME))

        




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--specified_save_path", type=str,default="/home/jwwang/URE_share/outcome/NL/clean_data", help="as named")
    parser.add_argument("--train_path", type=str,
                        default="/data/tywang/URE_share/data/tac/tac_pseudo_pos.pkl", help="as named")
    parser.add_argument("--dev_path", type=str,
                        default="/data/tywang/URE_share/data/tac/tac_pseudo_pos.pkl", help="as named")
    parser.add_argument("--config_path", type=str,
                        default="/home/jwwang/URE_UEE/NL/config/ace.json", help="as named")
    parser.add_argument("--sortedPlabelConfidence_path", type=str,
                        default="/home/jwwang/URE_UEE/NL/config/ace.json", help="as named")
    parser.add_argument("--cuda", type=int,
                    help="as named")
    parser.add_argument("--seed", type=int,
                    help="as named")
    args = parser.parse_args()

    NLNL_main(args)

    