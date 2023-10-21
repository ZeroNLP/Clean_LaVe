import copy
from os import error
import sys
import numpy as np
import torch



def dict_index(data:dict,index_arr):
    """
    遍历dict中的元素, 取出在index_arr中的下标
    """
    # 以数据的第一个key的value的len作为标准, 其余小于该len的key舍弃
    length = len(data[list(data.keys())[0]])

    data_copy = copy.deepcopy(data)
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

        # if isinstance(data_copy[k],torch.Tensor):
        #     data_copy[k] = torch.stack(data_copy[k])
        
    return data_copy

def dict_concate(dict1:dict,dict2:dict):
    assert dict1.keys()==dict2.keys()
    dict1_cp = copy.deepcopy(dict1)
    for k1 in dict1.keys():
        if isinstance(dict1[k1],list):
            dict1_cp[k1]+=dict2[k1]
        elif isinstance(dict1[k1],torch.Tensor):
            dict1_cp[k1] = torch.cat([dict1_cp[k1],dict2[k1]])
        else:
            print("error")
    return dict1_cp

def selectFromTextId(data:dict,text_id_arr:list,textid_key:str='text_id'):
    """
     select from a dict using key=text_id. key(text_id) is needed!
    :param data:
    :param text_id_arr:
    :param textid_key:
    :return: selected dict
    """
    data_copy = copy.deepcopy(data)
    L = len(data_copy[textid_key])
    index_arr = [idx for idx in range(L) if data_copy[textid_key][idx] in text_id_arr]  # 选出符合text_id_arr的index
    for k,_ in data_copy.items():
        # data_copy[k] = [data_copy[k][idx] for idx in index_arr]

        if isinstance(data_copy[k],np.ndarray):
            data_copy[k] = np.array([data_copy[k][idx] for idx in index_arr])
        elif isinstance(data_copy[k],torch.Tensor):
            data_copy[k] = data_copy[k][torch.tensor(index_arr).long()]
        else:
            #data_copy[k] = [data_copy[k][idx] for idx in range(L) if data_copy[textid_key][idx] in text_id_arr]
            data_copy[k] = [data_copy[k][idx] for idx in range(L) if idx in index_arr]
    return data_copy
    

def dict_add(dict1:dict,dict2:dict):
    assert dict1.keys()==dict2.keys()
    dict1_cp = copy.deepcopy(dict1)
    for k1 in dict1.keys():
        if isinstance(dict1[k1],list):
            dict1_cp[k1]+=dict2[k1]
        elif isinstance(dict1[k1],torch.Tensor):
            dict1_cp[k1] = torch.cat([dict1_cp[k1],dict2[k1]])
        elif isinstance(dict1[k1],np.ndarray):
            dict1_cp[k1] = np.concatenate([dict1_cp[k1],dict2[k1]])
        else:
            print("error")
            raise error
    return dict1_cp

def dict_sub(dict1:dict,dict2:dict,text_id_key:str='text_id'):
    """
    dict1 is a bigger dict
    text_id(key) is needed !!

    return dict: dict1-dict2
    """
    assert (text_id_key in dict1.keys()) and (text_id_key in dict2.keys())
    text_id1 = dict1[text_id_key]
    text_id2 = dict2[text_id_key]
    n = len(dict1[text_id_key])
    res_index = [i for i,textid1 in enumerate(text_id1) if textid1 not in text_id2]
    dict1_dict2 = dict_index(dict1,res_index)
    return dict1_dict2
    



if __name__=="__main__":
    pass
