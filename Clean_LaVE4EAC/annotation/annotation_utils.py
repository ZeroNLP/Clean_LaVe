import numpy as np
import pickle
import json
import copy
from sklearn.metrics import f1_score, precision_recall_fscore_support
import random
import torch


def dict_index(data:dict,index_arr):
    """
    遍历dict中的元素, 取出在index_arr中的下标
    """
    # 以数据的第一个key的value的len作为标准, 其余小于该len的key舍弃
    length = len(data[list(data.keys())[0]])

    data_copy = copy.copy(data)
    for k,_ in data_copy.items():
        l = len(data_copy[k])

        # data_copy[k] = [data_copy[k][idx] for idx in index_arr]

        if isinstance(data_copy[k],np.ndarray):
            data_copy[k] = np.array([data_copy[k][idx] for idx in index_arr])
        elif isinstance(data_copy[k],torch.Tensor):
            data_copy[k] = data_copy[k][torch.tensor(index_arr).long()]
        else: 
            data_copy[k] = [data_copy[k][idx] for idx in index_arr]

        
    return data_copy

def top_k_accuracy(output, labels, k=5, id2labels=None):
    """
    top_k_accuracy for whole output
    若id2labels非空, 则可返回topk的predict出来的relation
    """
    preds = np.argsort(output)[:, ::-1][:, :k]
    total = len(preds)
    right = 0
    predict_relations = []
    for l, p in zip(labels, preds):
        if id2labels is not None:
            predict_relations.append(id2labels[p[-1]])
        if l in p:
            right +=1
    if id2labels is not None:
        return right/total, predict_relations
    return right/total


def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def load_json(path):
    with open(path, "rt") as f:
        data = json.load(f)
    return data

def f1_score_(labels, preds, n_labels=42):
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average="micro")



def f1_score_help(gold, pred):
    cnt_pred = 0
    cnt_right_pred, cnt_right_rec = 0, 0
    cnt_gold = 0
    theta = 0.0001
    for single_gold, single_pred in zip(gold, pred):
        cnt_pred += len(single_pred)
        cnt_gold += len(single_gold)
        cnt_right_pred += len([element for element in single_pred if element in single_gold])
        cnt_right_rec += len([element for element in single_gold if element in single_pred])
    
    pre = cnt_right_pred/(cnt_pred + theta)
    rec = cnt_right_rec/(cnt_gold + theta)
    f1 = 2* pre * rec / (pre + rec + theta)

    return pre, rec, f1

    

def precision_recall_fscore_(labels, preds, n_labels=42):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average="micro")
    return p, r, f


def find_uppercase(str1:list, str2:list):
	"""
		str1 is a longer text
		str2(only lower case) is text included in str1
		find text in str1 that consist with str2, ignoring case. Then return the text in str1
	"""
	l2 = len(str2)
	for i in range(len(str1[:-l2])+1):
		if ' '.join(str1[i:i+l2]).lower().split()==str2:
			return ' '.join(str1[i:i+l2])
	return ' '.join(str2)

def apply_threshold(output, threshold=0.0, labels2id=None):
    output_ = output.copy()
    
    labels = np.argmax(output_, -1)
    confidence = np.max(output_, -1)

    # pick_index =[]
    # for i in range(23):
    #     index_i = np.where(labels==i)[0]
    #     sorted_index = sorted(index_i, key = lambda x: confidence[x], reverse=True)
    #     pick_index.extend(sorted_index[0:min(200, len(sorted_index))])
    # for idx in range(len(labels)):
    #     if idx not in pick_index:
    #         labels[idx] = labels2id["no_relation"]


    # print('max confidence: {}'.format(max(confidence)))
    mask_idx = np.where(confidence < threshold)[0]
    # print('len mask idx: {}'.format(len(mask_idx)))
    for idx in mask_idx:
         labels[idx] = labels2id["no_relation"]

    return labels


def find_optimal_threshold(labels: list, output, granularity=1000, labels2id=None):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    pres = []
    recs = []
    f1s = []
    cnt_0s = []
    for t in thresholds:
        preds = apply_threshold(output, threshold=t, labels2id=labels2id)
        pre, rec, f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        labels, preds, average="micro", labels=list(range(1, len(labels2id.keys())))) 
        
        cnt_0 = 0
        cnt_all = 0
        for label, pred in zip(labels, preds):
             if pred != 0:
                  cnt_all +=1
                  if label == 0:
                       cnt_0 +=1
        
        cnt_0s.append(cnt_0/(cnt_all + 0.0008))
        values.append(f1)  # dynamic???
        pres.append(pre)
        recs.append(rec)
        f1s.append(f1)

    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]
    # for t, p, r, f1, cnt_0 in zip(thresholds, pres, recs, f1s, cnt_0s):
    #      print("{}=> p:{}, r:{}, f1:{}, cnt_0: {}".format(t, p, r, f1, cnt_0))

    return best_threshold, values[best_metric_id]
