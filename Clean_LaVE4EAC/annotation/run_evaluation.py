import sys
import os
from pathlib import Path
import math

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))
import time
TIME=time.strftime("%m-%d-%H*%M*%S", time.localtime())
print("time",TIME)

import json

import numpy as np
import torch 
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from mnli import NLIEAEClassifierWithMappingHead, EAEInputFeatures
import arguments
from find_threshold import get_optimal_threshold
from annotation_utils import find_optimal_threshold, apply_threshold,set_global_random_seed, f1_score_help
from annotation_utils import find_uppercase,top_k_accuracy,dict_index
from utils.clean import get_format_train_text
from utils.pickle_picky import *
import warnings
warnings.filterwarnings("ignore")
from collections import Counter


set_global_random_seed(arguments.seed)  # 设置随机种子
# 加载config文件
with open(arguments.config_path, "rt") as f:
    config = json.load(f)



# 读取ace的数据
"""
    读取完数据会得到:
    features: List[REInputFeatures],
    events: List[int],
    labels: List[str],
"""

for configuration in config:
    labels2id = pickle_load(configuration['label2id'])
    id2labels = dict(zip(
        list(labels2id.values()),
        list(labels2id.keys())
    ))

    with open(arguments.run_evaluation_path, 'rt') as f:
        features, labels, events = [], [], []
        for line in f.readlines():
            lineData = json.loads(line)

            for event in lineData['event_mentions']:
                for argument in event['arguments']:
                    labels.append(labels2id[argument['role']])

                    features.append(
                        EAEInputFeatures(
                            lineData['text'],
                            lineData['text_piece'],

                            event['trigger'],
                            event['trigger_be'],
                            
                            argument['arg'],
                            argument['arg_be'],

                            argument['role'],
                            event['event_type'],
                            argument['type'],
                            "{}:{}".format(event['event_type'], argument['type'])
                        )
                    )


    n_labels = len(configuration['labels'])
    _ = configuration.pop("negative_threshold", None)
    classifier = NLIEAEClassifierWithMappingHead(**configuration)
    if arguments.output_path == '':
        output,template_sorted = classifier(
            features,
            batch_size=configuration["batch_size"]
        )
        pickle_save(output, os.path.join(arguments.generate_data_save_path, f"time{TIME}_output.pkl"))
    else:
        print("load output from {}".format(arguments.output_path))
        output = pickle_load(arguments.output_path)


    if arguments.threshold == -1:
        optimal_threshold = get_optimal_threshold(configuration)
    else:
        optimal_threshold = arguments.threshold

    print("use threshold:",optimal_threshold)
    outputs_filter = apply_threshold(output, threshold=optimal_threshold, labels2id = labels2id)

    pre, rec, f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        labels, outputs_filter, average="micro", labels=list(range(1, len(labels2id.keys()))))
    
    macro_pre, macro_rec, macro_f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        labels, outputs_filter, average="micro", labels=list(range(1, len(labels2id.keys()))))
    
    print("*"*20)
    print("pre:",macro_pre)
    print("rec:",macro_rec)
    print("f1:",f1)
    print("*"*20)
    
    
    
    # # 保存标注的数据
    if arguments.generate_data:
        label2id = pickle_load(configuration['label2id'])

        id2label = dict(zip(label2id.values(),label2id.keys()))
        dataset = {
        'text':[],
        'text_piece': [],

        'trigger':[],
        'trigger_be': [],
        'trigger_type': [],

        'argument':[],
        'argument_be': [],

        'role': [],
        'role_id': [],
        'output': [],
        'output_id': []
        }

        for feat, label, output in zip(features, labels, outputs_filter):
            feat:EAEInputFeatures
            dataset['text'].append(feat.context)
            dataset['text_piece'].append(feat.text_piece)

            dataset['trigger'].append(feat.trigger)
            dataset['trigger_be'].append(feat.trigger_be)
            dataset['trigger_type'].append(feat.triggger_type)

            dataset['argument'].append(feat.argument)
            dataset['argument_be'].append(feat.argument_be)

            dataset['role'].append(feat.role)
            dataset['role_id'].append(label2id[feat.role])
            dataset['output_id'].append(output)
            dataset['output'].append(id2label[output])


        print("*"*30)
        print("start to generate inferred data...")
        pos_index = [i for i in range(len(dataset['output_id'])) if dataset['output_id'][i] != 0]  # 选出positive数据的下标
        dataset_pos = dict_index(dataset,pos_index)
        print("tac selected num:{}".format(len(dataset_pos['output_id'])))
        print(Counter(dataset_pos['role_id']))
        print(len(Counter(dataset_pos['role_id'])))
        # print("ratio: {}".format(Counter(dataset_pos['role_id']).get(0)/len(dataset_pos['role_id'])))
        
        # if arguments.mode != "dev":
        # 那么就保存pseudo-positive的数据
        save_path = os.path.join(arguments.generate_data_save_path,"{}_{}_pos_T{}_f1{:.4f}.pkl".format(arguments.dataset,arguments.mode,TIME,f1))
        pickle_save(dataset_pos,save_path)
        print("[TRAIN]: save next-step data into {}".format(save_path))

        # 全集
        save_path = os.path.join(arguments.generate_data_save_path,"{}_MODE{}_whole_T{}_F1{:.4f}.pkl".format(arguments.dataset,arguments.mode,TIME,f1))
        pickle_save(dataset,save_path)
        print("save generate data into {}".format(save_path))
