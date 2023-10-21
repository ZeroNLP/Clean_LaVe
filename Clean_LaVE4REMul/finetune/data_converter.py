
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

from NL.NL import PlabelWithConfidenceItem
from dataclasses import dataclass
import random
from collections import defaultdict

@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int
labels2id = {"entailment": 0, "neutral": 1, "contradiction": 2}



def event2mnli_neutral(
    context,
    label, # p_label(id),
    subj, obj,
    label2template,
    negn=1,
    posn=1,
):
    """
        pos:
            entailment: 自己的template
            neutral: 别的pos的template
            contraction:  a and b has no relation 模板
    """

    mnli_instances = []

    # print(label2template)
    # print(label)
    # 生成entailment的label
    positive_template = random.choices(label2template[label], k=posn)  # 取出relation对应的模板
    mnli_instances.extend(  # 用这个rel对应的pos_template 来陈述 MNLIInputFeatures
        [
            MNLIInputFeatures(
                premise=context,
                hypothesis=f"{t.format(subj=subj, obj=obj)}.",
                label=labels2id["entailment"],
            )
            for t in positive_template
        ]
    )

    # 生成neutral的label
    canbe_selected_sorted_template = []
    for label_, template_ in label2template.items():
        if label_ != label:
            canbe_selected_sorted_template.extend(template_)
    negative_template = random.choices(canbe_selected_sorted_template, k=negn)
    mnli_instances.extend(
        [
            MNLIInputFeatures(
                premise=context,
                hypothesis=f"{t.format(subj=subj, obj=obj)}.",
                label=labels2id["neutral"],
            )
            for t in negative_template
        ]
    )

    # contradiction的label
    mnli_instances.append(
        MNLIInputFeatures(
            premise=context,
            hypothesis="{subj} and {obj} are not related.".format(subj=subj, obj=obj),
            label=labels2id["contradiction"],
        )
    )

    return mnli_instances


def prepare4mnil(plabelWithConfidences: list, initialData: dict, ratio: int, label2template: dict):
    selectData = plabelWithConfidences[0 : int(ratio * len(plabelWithConfidences))]

    # cal acc
    cnt_all, cnt_right = 0, 0
    for item in selectData:
        cnt_all+=1
        if isinstance(item, int):
            if initialData['label'][item] == initialData['top1'][item]:
                cnt_right+=1
        else:
            if initialData['label'][item.dataIndex] == initialData['top1'][item.dataIndex]:
                cnt_right+=1
    print("prepare data, acc: {}".format(cnt_right/cnt_all))

    # convert2mnil
    if isinstance(selectData[0], int):
        context = [initialData['text'][item] for item in selectData]
        label = [initialData['top1'][item] for item in selectData]
        subj = [initialData['subj'][item] for item in selectData]
        obj = [initialData['obj'][item] for item in selectData]
    else:
        context = [initialData['text'][item.dataIndex] for item in selectData]
        label = [initialData['top1'][item.dataIndex] for item in selectData]
        subj = [initialData['subj'][item.dataIndex] for item in selectData]
        obj = [initialData['obj'][item.dataIndex] for item in selectData]
    mnilDatas = []
    for singleContext, singleLabel, subj, obj in zip(context, label, subj, obj):
        mnilDatas.extend(event2mnli_neutral(singleContext, singleLabel,
                                            subj, obj,
                                             label2template))
    
    random.shuffle(mnilDatas)
    return mnilDatas


def prepare4mnil_count(plabelWithConfidences: list, initialData: dict, count: int, label2template: dict):
    selectData = plabelWithConfidences[0 : count]

    # cal acc
    cnt_all, cnt_right = 0, 0
    for item in selectData:
        cnt_all+=1
        if isinstance(item, int):
            if initialData['label'][item] == initialData['top1'][item]:
                cnt_right+=1
        else:
            if initialData['label'][item.dataIndex] == initialData['top1'][item.dataIndex]:
                cnt_right+=1
    print("prepare data, acc: {}".format(cnt_right/cnt_all))

    # convert2mnil
    if isinstance(selectData[0], int):
        context = [initialData['text'][item] for item in selectData]
        label = [initialData['top1'][item] for item in selectData]
        subj = [initialData['subj'][item] for item in selectData]
        obj = [initialData['obj'][item] for item in selectData]
    else:
        context = [initialData['text'][item.dataIndex] for item in selectData]
        label = [initialData['top1'][item.dataIndex] for item in selectData]
        subj = [initialData['subj'][item.dataIndex] for item in selectData]
        obj = [initialData['obj'][item.dataIndex] for item in selectData]
    mnilDatas = []
    for singleContext, singleLabel, subj, obj in zip(context, label, subj, obj):
        mnilDatas.extend(event2mnli_neutral(singleContext, singleLabel,
                                            subj, obj,
                                             label2template))
    
    random.shuffle(mnilDatas)
    return mnilDatas



def prepare4mnil_gold(initialData: dict, ratio: int, label2template: dict):
    selectData = random.sample(range(len(initialData['text'])), int(ratio * len(initialData['text'])))

    # convert2mnil

    context = [initialData['text'][item] for item in selectData]
    label = [initialData['top1'][item] for item in selectData]
    subj = [initialData['subj'][item] for item in selectData]
    obj = [initialData['obj'][item] for item in selectData]
    
    mnilDatas = []
    for singleContext, singleLabel, subj_, obj_ in zip(context, label, subj, obj):
        mnilDatas.extend(event2mnli_neutral(singleContext, singleLabel,
                                            subj_, obj_,
                                             label2template))
    
    random.shuffle(mnilDatas)
    return mnilDatas



def prepare4mnil_dynamic(plabelWithConfidences: list, initialData: dict, ratio: int, label2template: dict, num:int):
    selectData = plabelWithConfidences[0 : int(ratio * len(plabelWithConfidences))]
    otherData = plabelWithConfidences[int(ratio * len(plabelWithConfidences)): -1]

    # cal acc
    cnt_all, cnt_right = 0, 0
    for item in selectData:
        cnt_all+=1
        if isinstance(item, int):
            if initialData['label'][item] == initialData['top1'][item]:
                cnt_right+=1
        else:
            if initialData['label'][item.dataIndex] == initialData['top1'][item.dataIndex]:
                cnt_right+=1
    print("prepare data, acc: {}".format(cnt_right/cnt_all))

    # convert2mnil
    if isinstance(selectData[0], int):
        context = [initialData['text'][item] for item in selectData]
        label = [initialData['top1'][item] for item in selectData]
        subj = [initialData['subj'][item] for item in selectData]
        obj = [initialData['obj'][item] for item in selectData]
    else:
        context = [initialData['text'][item.dataIndex] for item in selectData]
        label = [initialData['top1'][item.dataIndex] for item in selectData]
        subj = [initialData['subj'][item.dataIndex] for item in selectData]
        obj = [initialData['obj'][item.dataIndex] for item in selectData]

    # step2:
    cntDict = defaultdict(int)
    indexByClass = defaultdict(list)
    if isinstance(plabelWithConfidences[0], int):
        for index in plabelWithConfidences:
            cntDict[initialData['top1'][index]] +=1
        for index in otherData:
            indexByClass[initialData['top1'][index]].append(index)
    else:
        for item in plabelWithConfidences:
            cntDict[initialData['top1'][item.dataIndex]] +=1
        for item in otherData:
            indexByClass[initialData['top1'][item.dataIndex]].append(item.dataIndex)
    
    for k, v in cntDict.items():
        ratio = v / len(plabelWithConfidences)
        context.extend([initialData['text'][item] for item in indexByClass[k][0: min(len(indexByClass[k]), int(ratio * num))]])
        label.extend([initialData['top1'][item] for item in indexByClass[k][0: min(len(indexByClass[k]), int(ratio * num))]])
        subj.extend([initialData['subj'][item] for item in indexByClass[k][0: min(len(indexByClass[k]), int(ratio * num))]])
        obj.extend([initialData['obj'][item] for item in indexByClass[k][0: min(len(indexByClass[k]), int(ratio * num))]])
        


    mnilDatas = []
    for singleContext, singleLabel, subj, obj in zip(context, label, subj, obj):
        mnilDatas.extend(event2mnli_neutral(singleContext, singleLabel,
                                            subj, obj,
                                             label2template))
    
    random.shuffle(mnilDatas)
    return mnilDatas