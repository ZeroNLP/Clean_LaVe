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

CLASSIFIERS = {"mnli-mapping": NLIRelationClassifierWithMappingHead}


def get_optimal_threshold():
    
    print("*"*20)
    print(" "*2,"find 0.01 threshold")
    print("*"*20)
    set_global_random_seed(arguments.seed)  # 设置随机种子
    run_evaluation_path_temp = arguments.run_evaluation_path
    arguments.run_evaluation_path = arguments.run_evaluation_path.replace("{}.json".format(arguments.mode),"0.01dev.json")
    mode_temp = arguments.mode
    outputs_temp = arguments.outputs
    arguments.outputs = None
    arguments.mode = "0.01dev"
    basic=False
    with open(arguments.config_path, "rt") as f:
        config = json.load(f)

    # 下面的事情只有tac会干
    if arguments.dataset=="tac":
        labels2id = (
            {label: i for i, label in enumerate(TACRED_LABELS)}
        )
        # id2labels
        id2labels = dict(zip(
            list(labels2id.values()),
            list(labels2id.keys())
        ))

        with open(arguments.run_evaluation_path, "rt") as f:  #输出features, labels, relations,subj_pos,obj_pos
            print("eval path:",arguments.run_evaluation_path)
            features, labels, relations,subj_pos,obj_pos = [], [],[],[],[]
            for line in json.load(f):
                id = line['id']
                # if arguments.split and arguments.selected_ratio is None:
                #     if id not in split_ids:
                #         continue
                line["relation"] = (
                    line["relation"] if not basic else TACRED_BASIC_LABELS_MAPPING.get(line["relation"], line["relation"])
                )
                subj_posistion= [line["subj_start"] , line["subj_end"]]
                subj_pos.append(subj_posistion)
                obj_posistion= [line["obj_start"] , line["obj_end"]]
                obj_pos.append(obj_posistion)
                features.append(
                    REInputFeatures(
                        subj=" ".join(line["token"][line["subj_start"] : line["subj_end"] + 1])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                        obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                        pair_type=f"{line['subj_type']}:{line['obj_type']}",
                        context=" ".join(line["token"])
                        .replace("-LRB-", "(")
                        .replace("-RRB-", ")")
                        .replace("-LSB-", "[")
                        .replace("-RSB-", "]"),
                        label=line["relation"],
                    )
                )
                relations.append(line["relation"])
                labels.append(labels2id[line["relation"]])

        # dict_keys(['text', 'rel', 'subj', 'obj', 'subj_type', 'obj_type', 'top1', 'top2', 'label', 'pos_or_not'])

    # print(unqualified)
    labels = np.array(labels)  # feature的label
    print("distribution of relations",Counter(relations))


    for configuration in config:
        n_labels = len(config[0]['labels'])
        _ = configuration.pop("negative_threshold", None)
        classifier = CLASSIFIERS[configuration["classification_model"]](negative_threshold=0.0, **configuration)
        output,template_socre,template_sorted, template2label = classifier(
            features,
            batch_size=configuration["batch_size"],
            multiclass=configuration["multiclass"],
        )
        optimal_threshold, _ = find_optimal_threshold(labels, output)  
        ignore_neg_pred =True
        top1,applied_threshold_output,_ = apply_threshold(output, threshold=optimal_threshold,ignore_negative_prediction=ignore_neg_pred)


        pre, rec, f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
            labels, top1, average="micro", labels=list(range(1, n_labels)) if arguments.dataset=="tac" else None
        )
        print("pre:",pre)
        print("rec:",rec)
        print("f1:",f1)
    arguments.mode = mode_temp
    arguments.outputs = outputs_temp
    arguments.run_evaluation_path = run_evaluation_path_temp
    set_global_random_seed(arguments.seed)
    return optimal_threshold
