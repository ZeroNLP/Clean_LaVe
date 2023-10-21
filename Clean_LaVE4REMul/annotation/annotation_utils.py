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
        if l!=length:
            print()
            print()
            print("***************************************")
            print("*       key:{} len is {}, skip!       *".format(k, l))
            print("***************************************")
            print()
            print()
            continue

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


def save(obj, path_name):
    print("保存到 ",path_name)
    with open(path_name, 'wb') as file:
        pickle.dump(obj, file)
def load_json(path):
    with open(path, "rt") as f:
        data = json.load(f)
    return data

def load(path_name: object) -> object:
    with open(path_name, 'rb') as file:
        return pickle.load(file)

def f1_score_(labels, preds, n_labels=42):
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average="micro")


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

def apply_threshold(output, threshold=0.0, ignore_negative_prediction=True):
    """Applies a threshold to determine whether is a relation or not"""
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:, 0] = 0.0
    activations = (output_ >= threshold).sum(-1).astype(np.int)  # 如果没有一个pos rel的 prob>threshold  , 那么归为no-rel
    output_[activations == 0, 0] = 1.00

    
    applied_threshold_output = copy.deepcopy(output_)
    score = np.max(output_,-1)
    return output_.argmax(-1),applied_threshold_output,score  # matrix


def find_optimal_threshold(labels, output, granularity=1000, metric=f1_score_):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds,_,_ = apply_threshold(output, threshold=t)
        values.append(metric(labels, preds))

    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]

    return best_threshold, values[best_metric_id]
