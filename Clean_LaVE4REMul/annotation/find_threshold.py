import sys
import os
import time
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
CURR_TIME=time.strftime("%m%d%H%M%S", time.localtime())
print("current time:",CURR_TIME)
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))

import arguments
from mnli import NLIRelationClassifierWithMappingHead, REInputFeatures
from tacred import *
# from utils.dict_relate import dict_index
from annotation_utils import find_optimal_threshold, apply_threshold,load,save,set_global_random_seed
from annotation_utils import find_uppercase
import json
from collections import Counter
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support
from utils.pickle_picky import *

# CLASSIFIERS = {"mnli-mapping": NLIRelationClassifierWithMappingHead}


def get_optimal_threshold(classifier):
    
    print("*"*20)
    print(" "*2,"find 0.01 threshold")
    print("*"*20)
    set_global_random_seed(arguments.seed)  # 设置随机种子
    run_evaluation_path_temp = arguments.run_evaluation_path
    arguments.run_evaluation_path = arguments.run_evaluation_path.replace("{}.json".format(arguments.mode),"dev.json")
    mode_temp = arguments.mode
    outputs_temp = arguments.outputs
    arguments.outputs = None
    arguments.mode = "dev"
    basic=False
    with open(arguments.config_path, "rt") as f:
        config = json.load(f)[0]

    # 下面的事情只有tac会干
    labels2id = pickle_load(config['label2id_path'])
    # id2labels
    id2labels = dict(zip(
        list(labels2id.values()),
        list(labels2id.keys())
    ))

    # 读取json文件
    with open(arguments.run_evaluation_path, "rt") as f:  
        features, labels, relations, initText = [], [],[], []
        for line in f.readlines():
            line = json.loads(line)
            id = line['id']
            features.append(
                REInputFeatures(
                    subj=line['entity_1'],
                    obj=line['entity_2'],
                    pair_type=f"TODO",
                    context=line['pure_text'],
                    label=line["label"],
                )
            )
            relations.append(line["label"])
            labels.append(labels2id[line["label"]])
            initText.append(line['text'])

    labels = np.array(labels)  # feature的label
    print("distribution of relations",Counter(relations))



    configuration = config
    n_labels = len(config['labels'])
    _ = configuration.pop("negative_threshold", None)
    # classifier = CLASSIFIERS[configuration["classification_model"]](negative_threshold=0.0, **configuration)
    output,template_socre,template_sorted, template2label = classifier(
        features,
        batch_size=configuration["batch_size"],
        multiclass=configuration["multiclass"],
    )
    optimal_threshold, _ = find_optimal_threshold(labels, output)  
    ignore_neg_pred =True
    top1,applied_threshold_output,_ = apply_threshold(output, threshold=optimal_threshold,ignore_negative_prediction=ignore_neg_pred)


    pre, rec, f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        labels, top1, average="micro", labels=list(range(1, n_labels))
    )
    print("[FIND T] pre:",pre)
    print("[FIND T] rec:",rec)
    print("[FIND T] f1:",f1)

    
    arguments.mode = mode_temp
    arguments.outputs = outputs_temp
    arguments.run_evaluation_path = run_evaluation_path_temp
    set_global_random_seed(arguments.seed)
    return optimal_threshold
