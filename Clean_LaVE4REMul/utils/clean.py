
import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))



import copy
import re
import random
import string
from tqdm import tqdm 
from distutils.log import error

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



def wiki_check_punctuation(text:str,punc=(string.punctuation+'–¡„“')):
    """
        wiki专用, 再标点前后加个空格
    """
    # punc+='–¡„“'
    # 'actien-gesellschaft „neptun“ schiffswerft und maschinenfabrik'
    text_new = ""
    for _,symbol in enumerate(text):
        if symbol in punc:
            text_new+=(" "+symbol+" ")
        else:
            text_new+=symbol
    text_new = ' '.join(text_new.split())
    text_new = text_new.replace('" -','"-').replace('. . .','...')
    return text_new


def replace_s(text,ori,target):
    ori = wiki_check_punctuation(ori)
    insensitive = re.compile(re.escape(ori), re.IGNORECASE)
    res= insensitive.sub(target, text,1)
    return res

def replace_s_tac(text,ori,target):
    insensitive = re.compile(re.escape(ori), re.IGNORECASE)
    res= insensitive.sub(target, text,1)
    return res

def replace_s_1(text,ori,target):
    # text = text.lower()
    # ori = ori.lower()
    index = -1
    try:
        text_sp = text.lower().split()
        ori_sp = ori.lower().split()
        L_ori = len(ori_sp)
        index = -1
        for i in range(len(text_sp)):
            if text_sp[i:i+L_ori]==ori_sp:
                index = i
                break
        res = text.split()[:index]+target.split()+text.split()[index+L_ori:]
        res = ' '.join(res)
        if index==-1:
            # print("======"*10)
            # print(text)
            # print(ori)
            # print(target)
            # return text.replace(ori,target)
            return re.sub(ori, target, text, flags=re.IGNORECASE) # 忽略大小写替换
    except:
        # print("======"*10)
        # print(text)
        # print(ori)
        # print(target)

        # res = text.replace(ori,target)
        res = re.sub(ori, target, text, flags=re.IGNORECASE)
    return res





def eliminate_noRelation(data, rel_key='rel'):
    not_norelation_index = []
    for i, rel in enumerate(data[rel_key]):
        if rel != 'no_relation':
            not_norelation_index.append(i)
    for k, _ in data.items():
        data[k] = [data[k][val_idx] for val_idx in not_norelation_index]
    return data
def text_adjust(text):
    """
    e.g.
    noumérat – moufdi zakaria airport
    noumérat–moufdi zakaria airport
    the same 
    """
    # text = text.lower()
    #text = text.replace('-',' - ')  # this for wiki80
    return text

def assure_replace(text:str):
    """
        查找一个text中有没有subj, obj 标志
    """
    if (not text.find("<S:")>=0) and (not text.find("<O:")>=0):
        return -3
    if not text.find("<S:")>=0:
        return -1
    if not text.find("<O:")>=0:
        return -2
    return 1
def obj_prefix(obj, obj_type): return " <O:{}> {} </O:{}> ".format(
    obj_type, obj, obj_type)

def subj_prefix(obj, obj_type): return " <S:{}> {} </S:{}> ".format(
    obj_type, obj, obj_type)

def get_format_train_text_tac(data: dict,text_k="text", subj_k="subj",
                          obj_k="obj", subj_t_k="subj_type", obj_t_k="obj_type",subj_pos = "subj_pos", obj_pos = "obj_pos"):
    """
    优先找出obj
    """
    index = 0
    formatted_texts = []
    all_prefix = set()
    for text, obj, subj, obj_t,subj_t,obj_p,subj_p  in tqdm(zip(data[text_k], data[obj_k], data[subj_k],
                                                            data[obj_t_k], data[subj_t_k],
                                                            data[obj_pos], data[subj_pos]),total=len(data[text_k])):
        origin_text = copy.deepcopy(text)
        # 优先替换obj
        text = text.split()
        obj_before = text[:obj_p[0]]
        obj_text = text[obj_p[0]:obj_p[1]+1]
        assert obj_text==obj.split()
        obj_text = obj_prefix(" ".join(obj_text),obj_t).split()
        obj_after = text[obj_p[1]+1:]
        text = ' '.join(obj_before+obj_text+obj_after)
        # 下面开始产生 subj的annotation
        if max(subj_p)<=min(obj_p):  # 精确
            text = text.split()
            subj_before = text[:subj_p[0]]
            subj_text = text[subj_p[0]:subj_p[1]+1]
            assert subj_text==subj.split()
            subj_text = subj_prefix(" ".join(subj_text),subj_t).split()
            subj_after = text[subj_p[1]+1:]
            text = ' '.join(subj_before+subj_text+subj_after)
        elif max(obj_p)<=min(subj_p): # 精确
            text = text.split()
            subj_before = text[:subj_p[0]+2]
            subj_text = text[subj_p[0]+2:subj_p[1]+2+1]
            assert subj_text==subj.split()
            subj_text = subj_prefix(" ".join(subj_text),subj_t).split()
            subj_after = text[subj_p[1]+2+1:]
            text = ' '.join(subj_before+subj_text+subj_after)
        else: # 可能不太精确, 但是可能是最好的方法了
            text = replace_s_tac(text,subj,subj_prefix(subj, subj_t))
        assert assure_replace(text)>0
        # 确保存在subj和obj的tag
        if assure_replace(text)<0:
            print(index,assure_replace(text))
        origin_text_1 = []
        # 确保能还原
        for w in text.split():
            w:str
            if not w.startswith(("</O:","<O:","</S:","<S:")):
                origin_text_1.append(w)
        origin_text_1 = " ".join(origin_text_1)
        assert origin_text_1==origin_text
        formatted_texts.append(text)
        all_prefix.add("<O:{}>".format(obj_t))
        all_prefix.add("<S:{}>".format(subj_t))
        all_prefix.add("</O:{}>".format(obj_t))
        all_prefix.add("</S:{}>".format(subj_t))
        index+=1
    return formatted_texts,all_prefix



def get_format_train_text(data: dict,mode:str,return_tag=False, text_k="text", subj_k="subj",
                          obj_k="obj", subj_t_k="subj_type", obj_t_k="obj_type",subj_pos = "subj_pos", obj_pos = "obj_pos"):
    """
    input:字典, key=['text', 'rel', 'subj', 'obj', 'subj_type', 'obj_type']
    text_k: text的key,其它类推 
    """
    # 句子前+<O:e1_type> 实体1<O:e1_type>+句子中间+<S:e2_type> 实体2<S:e2_type>+句子后

    res = copy.deepcopy(data)
    if mode=='wiki':
        text_formatted = []
        all_prefix = set() # all tag
        index = 0
        for text, obj, subj, obj_t,subj_t,  in tqdm(zip(data[text_k], data[obj_k], data[subj_k],
                                                data[obj_t_k], data[subj_t_k]),total=len(data[text_k])):
            text_f = replace_s(text,obj,obj_prefix(obj, obj_t))
            text_f = replace_s(text_f,subj,subj_prefix(subj, subj_t))
            if assure_replace(text_f)<0:
                print(index,assure_replace(text_f))
            all_prefix.add("<O:{}>".format(obj_t))
            all_prefix.add("<S:{}>".format(subj_t))
            all_prefix.add("</O:{}>".format(obj_t))
            all_prefix.add("</S:{}>".format(subj_t))
            text_formatted.append(text_f)
            index+=1
    elif mode=='tac':
        text_formatted,all_prefix = get_format_train_text_tac(data=data,text_k=text_k, subj_k=subj_k,
                          obj_k=obj_k, subj_t_k=subj_t_k, obj_t_k=obj_t_k,subj_pos = subj_pos, obj_pos = obj_pos)
    else:
        raise error
    
    res['text'] = text_formatted
    all_prefix = list(all_prefix)
    if return_tag:
        return res,all_prefix
    else:
        return res


if __name__ == "__main__":
    text = 'Jefferson J DeBlanc , a World War II fighter pilot who was awarded the Medal of Honor for shooting down five Japanese planes on a single day while running out of fuel , died Nov 22 in Lafayette , La'
    replace_s_tac(text,'Jefferson J DeBlanc',"<'Jefferson J DeBlanc'>")
    ########################################### test #####################################################
    test_cases = {
        'text':["hi, I went to the high school yesterday.","he went to the university yesterday."],
        'subj':['I','he'],
        'obj':['high school','university'],
        'subj_type':['PER','PER'],
        'obj_type':['LOC','LOC']
    }
    dict,tags = get_format_train_text(test_cases,return_tag=True)
    print(dict)
    print(tags)
    ########################################### test #####################################################


    # 这个是ok了应该
    # train = load("/root/tywang/URE/data/wiki80/train_10548.pkl")
    # dicts,tags = get_format_train_text(train,True)


    # texts = "hi, I and my mom went back home"
    # ori = ' back   '
    # target = ' <12345> sad sad sad sad sad sad '
    # res = replace_s(texts,ori,target)
    
    # dev = load("/root/tywang/URE/data/tacred/dev.pkl")
    # dev = eliminate_noRelation(dev)
    # test = load("/root/tywang/URE/data/tacred/test.pkl")
    # test = eliminate_noRelation(test)
    # rel2ids = load("/root/tywang/URE/data/tacred/rel2id.pkl")
    # print(rel2ids)
    # dev['label'] = rel2id(dev['rel'])
    # test['label'] = rel2id(test['rel'])
    # train['label'] = rel2id(train['rel'])
    # train['p_label'] = rel2id(train['p_rel'])
    # print(dev.keys())
    # print(test.keys())
    # print(train.keys())
    # print(dev['rel'][:20])
    # print(dev['label'][:20])
    # save(dev,"/root/tywang/URE/data/tacred/dev.pkl")
    # save(test,"/root/tywang/URE/data/tacred/test.pkl")
    # save(train,"/root/tywang/URE/data/tacred/train.pkl")

    # dev_res = get_format_train_text(dev)
    # test_res = get_format_train_text(test)
    # print(dev_res.keys())
    # dev['raw_text'] = dev['text']
    # test['raw_text'] = test['text']
    # dev['text'] = dev_res['text_formatted']
    # test['text'] = test_res['text_formatted']
    # print(dev.keys(),len(dev['text'])==len(dev['rel']))
    # print(test.keys(),len(test['text'])==len(test['rel']))
    # save(dev,"/root/tywang/URE/data/tacred/dev.pkl")
    # save(test,"/root/tywang/URE/data/tacred/test.pkl")
    # print(dev_res.keys())
    # print(test_res.keys())
    # pretrain = load("/root/tywang/URE/data/tacred_pretrain/pretrain_data_3b_1.pkl")
    # train = load("/root/tywang/URE/data/tacred/train.pkl")
    # # train['p_rel']
    # train['p_rel'] = [-1 for i in range(len(train['text']))]

    # print(train.keys())
    # print(train['text'][2])
    # texts = load("/root/tywang/code_for_infer_T5/data/tacred/train_41.pkl")
    # _,all_prefix = get_format_train_text(texts)
    # save(all_prefix,'/root/tywang/URE/data/tacred/tacred_e_tag/tacred_e_tag.pkl')
    # print(all_prefix)

    # for p_rel,tid in zip(pretrain['p_rel'],pretrain['textid']):
    #     train['p_rel'][tid] = p_rel

    # print(train['p_rel'][:20])
    # save(train,"/root/tywang/URE/data/tacred/train.pkl")
    # print(pretrain['textid'])

    # for text,text_f,subj,obj,subt,objt in zip(res['text'],res['text_formatted'],res['subj'],\
    #     res['obj'],res['subj_type'],res['obj_type']):
    #     if random.random()<0.05:
    #         print("==================")
    #         print(text)
    #         print(text_f)
    #         print(subj)
    #         print(obj)
    #         print(subt)
    #         print(objt)
