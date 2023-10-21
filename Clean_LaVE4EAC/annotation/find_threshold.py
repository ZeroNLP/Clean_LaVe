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
from mnli import NLIEAEClassifierWithMappingHead, EAEInputFeatures
from annotation_utils import find_optimal_threshold, apply_threshold,set_global_random_seed, f1_score_help
from annotation_utils import find_uppercase
import json
from collections import Counter
import numpy as np
from utils.pickle_picky import *
from sklearn.metrics import f1_score, precision_recall_fscore_support


def get_optimal_threshold(config):
    
    print("*"*20)
    print(" "*2,"find threshold--all dev")
    print("*"*20)
    dev_path = arguments.run_evaluation_path.replace("{}.json".format(arguments.mode),"dev.json")

    labels2id = pickle_load(config['label2id'])
    id2labels = dict(zip(
        list(labels2id.values()),
        list(labels2id.keys())
    ))
    
    with open(dev_path, 'rt') as f:
        features, labels = [], []
        for line in f.readlines():
            lineData = json.loads(line)

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

    print("distribution of relations",
            Counter(list(labels)))
    
    n_labels = len(config['labels'])
    classifier = NLIEAEClassifierWithMappingHead(**config)
    output, template_sorted = classifier(
        features,
        batch_size=config["batch_size"]
    )
    optimal_threshold, _ = find_optimal_threshold(labels, output, labels2id = labels2id)  
    outputs_filter = apply_threshold(output, threshold=optimal_threshold, labels2id = labels2id)


    pre, rec, f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        labels, outputs_filter, average="micro", labels=list(range(1, len(labels2id.keys()))))   
    
    print("pre:",pre)
    print("rec:",rec)
    print("f1:",f1)

    return optimal_threshold
