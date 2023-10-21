
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


@dataclass
class MNLIInputFeatures:
    premise: str
    hypothesis: str
    label: int
labels2id = {"entailment": 2, "neutral": 1, "contradiction": 0}



def event2mnli_neutral(
    context,
    label, # p_label(id),
    arg, tri, tri_type,
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
                hypothesis=t.format(arg=arg,trg=tri, trg_subtype=tri_type.split('.')[-1]),
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
                hypothesis=t.format(arg=arg,trg=tri, trg_subtype=tri_type.split('.')[-1]),
                label=labels2id["neutral"],
            )
            for t in negative_template
        ]
    )

    # contradiction的label
    mnli_instances.append(
        MNLIInputFeatures(
            premise=context,
            hypothesis="{arg} is not an argument of {trg}".format(
                arg=arg,trg=tri, trg_subtype=tri_type.split('.')[-1]
            ),
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
        if item.labelIndex == initialData['role_id'][item.dataIndex]:
            cnt_right+=1
    print("prepare data, acc: {}".format(cnt_right/cnt_all))

    # convert2mnil
    context = [initialData['text'][item.dataIndex] for item in selectData]
    label = [item.labelIndex for item in selectData]
    argument = [initialData['argument'][item.dataIndex] for item in selectData]
    trigger = [initialData['trigger'][item.dataIndex] for item in selectData]
    trigger_type = [initialData['trigger_type'][item.dataIndex] for item in selectData]
    mnilDatas = []
    for singleContext, singleLabel, arg, tri, tri_type in zip(context, label, argument, trigger, trigger_type):
        mnilDatas.extend(event2mnli_neutral(singleContext, singleLabel,
                                            arg, tri, tri_type,
                                             label2template))
    
    random.shuffle(mnilDatas)
    return mnilDatas


def prepare4mnil_gold(initialData: dict, ratio: int, label2template: dict):
    selectData = random.sample(range(len(initialData['text'])), int(ratio * len(initialData['text'])))

    # cal acc
    # cnt_all, cnt_right = 0, 0
    # for item in selectData:
    #     cnt_all+=1
    #     if item.labelIndex == initialData['role_id'][item]:
    #         cnt_right+=1
    # print("prepare data, acc: {}".format(cnt_right/cnt_all))

    # convert2mnil
    context = [initialData['text'][item] for item in selectData]
    label = [initialData['role_id'][item] for item in selectData]
    argument = [initialData['argument'][item] for item in selectData]
    trigger = [initialData['trigger'][item] for item in selectData]
    trigger_type = [initialData['trigger_type'][item] for item in selectData]
    mnilDatas = []
    for singleContext, singleLabel, arg, tri, tri_type in zip(context, label, argument, trigger, trigger_type):
        mnilDatas.extend(event2mnli_neutral(singleContext, singleLabel,
                                            arg, tri, tri_type,
                                             label2template))
    
    random.shuffle(mnilDatas)
    return mnilDatas
