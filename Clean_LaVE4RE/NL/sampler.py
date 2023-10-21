
import argparse
from cProfile import label
import sys
import os
from pathlib import Path
import time
TIME=time.strftime("%m%d%H%M%S", time.localtime())# 记录被初始化的时间

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

from utils.pickle_picky import load, save
import numpy as np
from collections import Counter, defaultdict
from utils.dict_relate import dict_index



# threadhold1=0.7401767373085022 #18
# threadhold1=0.6963825225830078 #17
# threadhold1=0.721817135810852 #16
def sampler(args):

    ratios = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.15, 0.2, 0.25, 0.3]

    if args.dataset=='tac':
        n_rel=42
        train_data=load(args.pos_annotate_path)
        train_data['p_label'] = train_data['top1']
        n_train = len(train_data['p_label'])

        label2id=load(args.label2id)
        id2label={}
        for key,value in label2id.items():
            id2label[value]=key
        # data=load('/home/jwwang/URE_share/outcome/NL/NL_tac_T0529130943.pkl')
        # data=load('/home/jwwang/URE_share/outcome/NL/clean_data/NLFiltered_tac_RT0.05_RT21.0_SD18.pkl')
        confidence_path=args.confidence_path
        p_label_confidence=load(confidence_path)
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]
    elif args.dataset=="wikifact":
        label2id=load('/home/jwwang/URE_share/data/tac/label2id.pkl')
        id2label={}
        for key,value in label2id.items():
            id2label[value]=key
        whole_data=load('/home/jwwang/URE_share/outcome/NL/wikifact4tac/wikifact4tac_whole.pkl')
        train_data={
            'text':[],
            'label':[],
            'p_label':[],
            'top1':[],
            'subj':[],
            'obj': [],
            'template': []
        }
        keyMap={
            'text':'text',
            'label': 'target',
            'p_label':'p_rel',
            'top1':'top1',
            'subj': 'subj',
            'obj': 'obj',
            'template': 'template'
        }
        p_label_confidence=[]
        for item in whole_data:
            item['p_rel']=label2id[item['p_rel']]
            item['target']=item['p_rel']
            for key in train_data.keys():
                train_data[key].append(item[keyMap[key]])
            p_label_confidence.append(item['confidence'])
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]
    elif args.dataset=="wiki":
        n_rel=80
        train_data=load(args.pos_annotate_path)
        print(train_data.keys())
        train_data['p_label'] = train_data['top1']
        n_train = len(train_data['p_label'])

        label2id=load(args.label2id)
        id2label={}
        for key,value in label2id.items():
            id2label[value]=key
        # data=load('/home/jwwang/URE_share/outcome/NL/NL_tac_T0529130943.pkl')
        # data=load('/home/jwwang/URE_share/outcome/NL/clean_data/NLFiltered_tac_RT0.05_RT21.0_SD18.pkl')
        confidence_path=args.confidence_path
        p_label_confidence=load(confidence_path)
        confidence_index = np.argsort(np.array(p_label_confidence))[::-1]


    threshold_confidence=p_label_confidence[confidence_index[int(0.01*n_train)]]
    print('=> 0.01part of train is [{}]'.format(threshold_confidence))


    indexes = {}
    select_num = [int(rt*n_train) for rt in ratios]
    accs = []
    for select_n,rt in zip(select_num,ratios):
        selected403  = confidence_index[:int(select_n)]
        indexes[rt]=selected403

        # 计算acc
        text= np.array([train_data['text'][index] for index in selected403])
        Slabel = np.array([train_data['label'][index] for index in selected403]) # ground-truth
        Sp_label = np.array([train_data['p_label'][index] for index in selected403]) # pseudo label
        counter=Counter()
        for item in Sp_label:
            counter[item]+=1
        counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
        # print('*'*50)
        # for key,val in counter:
        #     print('{}_{}'.format(id2label[key],val),end='|')
        # print()
        n_cate = len(set(Sp_label))
        acc = sum(Slabel==Sp_label)/len(Sp_label)
        accs.append(acc)
        # print("前{} {} confident 的数据 acc= {}, 类别数:{},confidence:{}".format(rt,select_n,acc,n_cate,p_label_confidence[selected403[-1]]))


    #! class-aware
    other_index = confidence_index[int(args.ratio * n_train):-1]
    class_based_num=defaultdict(int)
    class_based_sortIndex=defaultdict(list)
    for i in other_index:
        class_based_sortIndex[train_data['p_label'][i]].append(i)
    for item in train_data['p_label']:
        class_based_num[item]+=1

    selected403=[]
    for key,item in class_based_sortIndex.items():
        selected403.extend(item[:min(len(item), int(args.extend_num * (class_based_num[key]/len(train_data['p_label'])) ))])

        # #500 if int(rt*len(item))>500 else
        # for index0,index in enumerate(item):
        #     if p_label_confidence[index]>=threshold_confidence or (index0)/len(item)<0.5:
        #         selected403.append(index)

    selected403.extend(confidence_index[:int(args.ratio * n_train)])

    class_based_total=defaultdict(int)
    class_based_right=defaultdict(int)
    for index in selected403:
        class_based_total[train_data['p_label'][index]] += 1
        if train_data['p_label'][index] == train_data['label'][index]:
            class_based_right[train_data['p_label'][index]] += 1
    
    print('*'*50)
    for index in class_based_total.keys():
        print(f"\textend_num: {args.extend_num}, label: {index}, acc: {class_based_right[index]/ class_based_total[index]}, num: {class_based_total[index]}")
    print('*'*50)



    Slabel = np.array([train_data['label'][index] for index in selected403]) # ground-truth
    Sp_label = np.array([train_data['p_label'][index] for index in selected403]) # pseudo label
    counter=Counter()
    for item in Sp_label:
        counter[item]+=1
    counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
    # print('*'*50)
    # for key,val in counter:
    #     print('{}_{}'.format(id2label[key],val),end='|')
    # print()
    n_cate = len(set(Sp_label))
    acc = sum(Slabel==Sp_label)/len(Sp_label)
    accs.append(acc)
    print("final: ratio:{} 的数据 num= {}, acc= {}, 类别数:{}".format(args.ratio,len(selected403),acc,n_cate))

    index=selected403
    selected_data = dict_index(train_data,index)

    save(selected_data,args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=16, help="as named")
    parser.add_argument("--dataset", type=str, default='wiki', help="as named")
    parser.add_argument("--ratio", type=float, default=0.08, help="as named")
    parser.add_argument("--save_path", type=str, default='', help="as named")

    parser.add_argument("--pos_annotate_path", type=str, help="as named")
    parser.add_argument("--confidence_path", type=str, help="as named")
    parser.add_argument("--label2id", type=str, help="as named")

    parser.add_argument("--extend_num", type=int, help="as named")

    args=parser.parse_args()
    sampler(args)