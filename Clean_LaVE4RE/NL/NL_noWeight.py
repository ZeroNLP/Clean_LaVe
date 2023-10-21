
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
from utils.pickle_picky import load, save
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
from collections import defaultdict

class Train_dataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __getitem__(self, index):
        return {
            'text': self.data['text'][index],
            'p_label': self.data['p_label'][index],  # pseudo_label
            'index': self.data['index'][index],
        }

    def __len__(self):
        return len(self.data['text'])

def classBasedSampler(args,train_data:dict,threshold2,):
    class_based_index={}
    for i in range(args.n_rel):
        class_based_index[i]=[]
    for index,p_label in enumerate(train_data['p_label']):
        class_based_index[p_label].append(index)
    selected403=[]
    for key,item in class_based_index.items():
        selected403.extend(item[0:max(10,
        ( int(threshold2*len(item)))
        )])
    index=selected403
    selected_data = dict_index(train_data,index)
    acc = sum(np.array(selected_data['label'])==np.array(selected_data['top1']))/len(selected_data['label'])
    print("selected data filtered acc:{}, num:{}".format(acc,len(selected_data['text'])))
    return selected_data
    



def NLNL_main(args):
    print(args)
    set_global_random_seed(args.seed)
    if args.train_path.find("wiki")!=-1:
        args.dataset=mode = "wiki"
    else:
        args.dataset=mode = "tac"
    train_data = load(args.train_path)
    train_data['p_label'] = train_data['top1']


    print("*"*10,"information","*"*10)
    info_acc = sum(np.array(train_data['p_label'])==np.array(train_data['label']))/len(train_data['p_label'])
    n_pseudo_label_relation = len(set(train_data['p_label']))
    print("acc:{:.4f}".format(info_acc))
    print("n_relation:{}".format(n_pseudo_label_relation))
    print("N_data:{}".format(len(train_data['p_label'])))
    print("*"*10,"***********","*"*10)
    

    num_classes = args.n_rel
    args.N_train = N_train = len(train_data['text'])
    train_data['index'] = [i for i in range(args.N_train)]
    train_dataset = Train_dataset(train_data)
    train_loader = util_data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    inds_noisy = np.asarray([index for index in range(len(train_data['p_label'])) if train_data['p_label'][index]!=train_data['label'][index]  ])
    inds_clean = np.delete(np.arange(N_train), inds_noisy)
    tags = load(args.e_tags_path)
    device = torch.device('cuda:{}'.format(args.cuda_index))
    args.device = device
    bert_model = SentenceTransformer(args.model_dir)
    sccl_model:SCCL_BERT = SCCL_BERT(bert_model, args.max_len,
                           device, args.n_rel, True, tags).to(device)

    ##
    # set training related
    # net:NLNL_Net = NLNL_Net(sccl_model).to(device)
    optimizer = torch.optim.AdamW([
        {'params': sccl_model.sentbert.parameters()},
        {'params': sccl_model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    ##

    # 产生每个类的weight
    weight = torch.FloatTensor(num_classes).zero_() + 1.
    for i in range(num_classes):
        weight[i] = (torch.from_numpy(np.array(train_data['p_label']).astype(int)) == i).sum()  
    weight = 1 / (weight / weight.max()).to(device)



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
    train_preds = torch.zeros(N_train, num_classes) - 1.
    num_hist = 10
    train_preds_hist = torch.zeros(N_train, num_hist, num_classes)   # [45000,10,10]
    pl_ratio = 0.
    nl_ratio = 1.-pl_ratio  
    train_losses = torch.zeros(N_train) - 1.  # 每个数据的loss
    ##

    ##
    # train
    for epoch in range(args.epoch):
        train_loss = train_loss_neg = train_acc = 0.0
        pl = 0.; nl = 0.
        sccl_model.train()
        accs = []
        losses = []
        data_predict = torch.zeros(N_train,dtype=torch.long) # 预测值
        data_predict_confidence = torch.zeros(N_train) # 预测值的confidence
        for i, data in enumerate(train_loader):
            
            text, labels, index = data['text'],data['p_label'],data['index']
            labels_neg = (labels.unsqueeze(-1).repeat(1, args.ln_neg)
                      + torch.LongTensor(len(labels), args.ln_neg).random_(1, num_classes)) % num_classes
            assert labels_neg.max() <= num_classes-1
            assert labels_neg.min() >= 0
            assert (labels_neg != labels.unsqueeze(-1).repeat(1, args.ln_neg)
                    ).sum() == len(labels)*args.ln_neg  # 保证得到的都是和原来label不同的数据
            labels = labels.to(device)
            labels_neg = labels_neg.to(device)
            logits = sccl_model.out(sccl_model.get_embeddings_PURE(text))

            s_neg = torch.log(torch.clamp(1.-F.softmax(logits, -1), min=1e-5, max=1.))  # log(1-pk)
            if args.use_weight:
                s_neg *= weight[labels].unsqueeze(-1).expand(s_neg.size()).to(device) # dynamic weight
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

            train_loss += logits.size(0)*criterion(logits, labels).data
            train_loss_neg += logits.size(0) * criterion_nll(s_neg, labels_neg[:, 0]).data

            train_losses[index] = criterion_nr(logits, labels).cpu().data  # 记录这次每个数据的 CEloss
            

            labels = labels*0 - 100  #不使用PL, label=-100 可以使得Cross-entropy loss=0
            
            loss_neg = criterion_nll(s_neg.repeat(args.ln_neg, 1), labels_neg.t().contiguous().view(-1)) * float((labels_neg >= 0).sum())
            loss_pl = criterion(logits, labels)* float((labels >= 0).sum())
            
            loss = (loss_neg) / (float((labels >= 0).sum()) +float((labels_neg[:, 0] >= 0).sum()))
            loss.backward()
            optimizer.step()
            l = logits.size(0)*loss.detach().cpu().data
            train_loss+=l
            
            losses.append(l/logits.size(0))
            train_preds[index.cpu()] = F.softmax(logits, -1).cpu().data

            pl += float((labels >= 0).sum())
            print('\r', "EPOCH[{}] step {}/{} ,  loss: {:.4f} train_acc: {:.4f}  ".format(epoch+1,i+1,len(train_loader),np.mean(losses),np.mean(accs)), end='', flush=True)
            nl += float((labels_neg[:, 0] >= 0).sum())
            # if i==10:break


        ## 用于在tac fewshot的时候, 每类都选前k confident的数据.
        # select topk confident data of each category in predition 
        predicts = dict()
        for i,(pre,confi) in enumerate(zip(data_predict,data_predict_confidence)):
            pre = pre.item()
            confi = confi.item()
            if pre not in predicts:
                predicts[pre] = [(i,confi)] # (data_index, confidence)
            else:
                predicts[pre].append((i,confi))
        for k in predicts.keys():  # sort it 
            predicts[k].sort(key=lambda x : x[1],reverse=True)


        topks = [1,2,3]
        for topk in topks:
            selected_index = []
            for k in predicts.keys(): # 遍历所有的类别
                items  = [item[0] for item in predicts[k][:topk]]
                selected_index.extend(items) # 记录下所选数据的标签
            selected_label = [train_data['label'][i] for i in selected_index]
            # print(Counter(selected_label))
            selected_data = dict_index(train_data,selected_index)
            selected_data['top1'] = selected_data['p_label'] = selected_data['label'] # fewshot模式, 直接用label.
            # print(Counter(selected_data['top1']))
            #save(selected_data,"/home/tywang/URE-master/scripts/fewshot_33lab/tac_confi_select_k{}.pkl".format(topk))
        ##



        train_loss /= N_train
        train_loss_neg /= N_train
        train_acc /= N_train
        pl_ratio = pl / float(N_train)

        ## 记录confident
        assert train_preds[train_preds < 0].nelement() == 0
        train_preds_hist[:, epoch % num_hist] = train_preds
        train_preds = train_preds*0 - 1.
        assert train_losses[train_losses < 0].nelement() == 0
        train_losses = train_losses*0 - 1.
        ##

        #[dynamic weight]
        weightV2 = torch.FloatTensor(num_classes).zero_() + 1.
        ##  根据confidence选择clean数据
        # 计算仅仅由 本身confidence 排序得到的数据的acc
        p_label_confidence = train_preds_hist.mean(1)[torch.arange(N_train), np.array(train_data['p_label']).astype(int)] # shape = N_train
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]  # confidence从大到小排序
        indexes = []
        ratios = np.arange(0.01, 0.99, 0.01)
        total_acc = []
        
        n_train = len(train_dataset)
        select_num = [int(rt*n_train) for rt in ratios]
        for select_n,rt in zip(select_num,ratios):
            selected403  = confidence_index[:int(select_n)]
            indexes.append(selected403)
            # 计算acc
            Slabel = np.array([train_data['label'][index] for index in selected403]) # ground-truth
            Sp_label = np.array([train_data['p_label'][index] for index in selected403]) # pseudo label
            n_cate = len(set(Sp_label))
            acc = sum(Slabel==Sp_label)/len(Sp_label)
            # print("前{} {} confident 的数据 acc= {}, 类别数:{}".format(rt,select_n,acc,n_cate))
            total_acc.append(acc)
            if rt == 0.05:
                cnt_right = defaultdict(int)
                cnt_all = defaultdict(int)
                for i ,j in zip(Slabel, Sp_label):
                    cnt_all[j] += 1
                    if i==j:
                        cnt_right[j] += 1
                
                for k ,v in cnt_all.items():
                    print(f"\t=======> label id:{k}, acc:{cnt_right[k]/cnt_all[k]}")
            # if rt == 0.1:
            #     for p_label in Sp_label:
            #         weightV2[p_label] += 1
        print("*"* 10)
        print(total_acc)
        print("*"* 10)
        # #[dynamic weight]
        # weightV2 = (weightV2 / weightV2.max()).to(device)
        # weightV2 = torch.exp(0.5 - weightV2/torch.max(weightV2))
        # print(weightV2)
        # weight *= weightV2
        # print(weight)


        # if epoch+1==args.epoch:
        #     # select 数据
        #     if str(args.specified_save_path).endswith('wiki'):
        #         save(p_label_confidence,'/data/jwwang/URE/output/NL/wiki_seed{}_time{}_pconfidence.pkl'.format(args.seed, TIME))
        #     else:
        #         save(p_label_confidence,'/data/jwwang/URE/output/NL/tac_seed{}_time{}_pconfidence.pkl'.format(args.seed, TIME))
    

        # 画分布图
        if args.plot:
            clean_plot = train_preds_hist.mean(1)[torch.arange(N_train)[
                inds_clean], np.array(train_data['p_label']).astype(int)[inds_clean]]
            noisy_plot = train_preds_hist.mean(1)[torch.arange(N_train)[
                inds_noisy], np.array(train_data['p_label']).astype(int)[inds_noisy]]
            clean_plot = clean_plot.numpy()
            noisy_plot = noisy_plot.numpy()
            plt.hist(clean_plot, bins=33, edgecolor='black', alpha=0.5,
                    range=(0, 1), label='clean', histtype='bar')
            plt.hist(noisy_plot, bins=33, edgecolor='black', alpha=0.5,
                    range=(0, 1), label='noisy', histtype='bar')
            plt.xlabel('probability')
            plt.ylabel('number of data')
            plt.grid()
            plt.legend()
            img_dir = '/home/jwwang/URE_share/outcome/NL/imgs/{}_confidence_distribution_epoch{}_T{}.jpg'.format(mode,epoch,TIME)
            print("img here: ",img_dir)
            plt.savefig(img_dir)
            plt.clf()




if __name__ == "__main__":

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--cuda_index", type=int, default=2, help="as named")
    parser.add_argument('--lr', type=float, default=4e-7, help='learning rate')
    parser.add_argument('--lr_scale', type=int, default=100, help='as named')
    parser.add_argument('--epoch', type=int, default=10, help='as named')
    parser.add_argument("--specified_save_path", type=str,default="/home/jwwang/URE_share/outcome/NL/clean_data", help="as named")
    parser.add_argument('--threshold1', type=float, default=0.09, help='as named')
    parser.add_argument('--threshold2', type=float, default=0.5, help='as named')
    
    """wiki"""
    # parser.add_argument('--n_rel', type=int, default=80, help='as named')
    # parser.add_argument("--train_path", type=str,
    #                     default="/data/tywang/URE_share/data/wiki/train_annotated.pkl", help="as named")
    # parser.add_argument("--e_tags_path", type=str,
    #                     default="/data/tywang/URE_share/data/wiki/tags.pkl", help="形如<S:PERSON>的tag")
    # parser.add_argument("--save_dir", type=str,
    #                     default="/data/tywang/URE_share/NL/outputs", help="it is used to save imgs")
    # parser.add_argument('--ln_neg', type=int, default=80,
    #                     help='number of negative labels on single image for training, equal to n_rel')

    """tac"""
    parser.add_argument('--n_rel', type=int, default=41, help='as named') # 由于用于选clean的数据是pseudo_label=positive的数据, 因此只有41个类
    parser.add_argument("--train_path", type=str,
                        default="/data/tywang/URE_share/data/tac/tac_pseudo_pos.pkl", help="as named")
    parser.add_argument("--e_tags_path", type=str,
                        default="/data/tywang/URE_share/data/tac/tags.pkl", help="as named")
    parser.add_argument("--save_dir", type=str,
                        default="/data/tywang/URE_share/NL/outputs", help="as named")
    parser.add_argument('--ln_neg', type=int, default=41,
                        help='number of negative labels on single image for training (ex. 110 for cifar100)')
    
    parser.add_argument("--use_weight", type=str2bool,
                        default="False", help="as named")


    """communal"""

    parser.add_argument('--save_info', type=str,
                        default="", help='as named')
    parser.add_argument('--model_dir', type=str,
                        default='/data/transformers/bert-base-uncased', help='as named')
    parser.add_argument('--max_len', type=int, default=64,
                        help='length of input sentence')
    parser.add_argument('--plot', type=bool, default=False,
                        help='是否画分布图')
    args = parser.parse_args()

    print(args)

    NLNL_main(args)

    