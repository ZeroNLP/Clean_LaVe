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
import time
TIME=time.strftime("%m-%d-%H*%M*%S", time.localtime())

from collections import defaultdict
from typing import Dict, List
from dataclasses import dataclass
import arguments
import numpy as np
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)



from base import Classifier, np_softmax
from utils.pickle_picky import *

@dataclass
class EAEInputFeatures:
    context: str
    text_piece: list
    trigger: str
    trigger_be: str

    argument: str
    argument_be: str

    role: str
    
    triggger_type: str
    entity_type: str
    pair_type: str

class _NLIEAEClassifier(Classifier):
    def __init__(
            self,
            labels: List[str],
            *args,
            pretrained_model: str = arguments.model_path,
            use_cuda=True,
            half=False,
            verbose=True,
            negative_threshold=0.95,
            max_activations=np.inf,
            **kwargs,
        ):
            super().__init__(
                labels,
                pretrained_model=pretrained_model,
                use_cuda=use_cuda,
                verbose=verbose,
                half=half,
            )
            self.negative_threshold = negative_threshold
            self.max_activations = max_activations # 超过max数量的分类, 视为noisy sample
            self.n_rel = len(labels) 

            def idx2label(idx):
                return self.labels[idx]
            self.idx2label = np.vectorize(idx2label)
    
    def _initialize(self, pretrained_model):
        print("used model from:",pretrained_model)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        if arguments.load_dict:
            print("load weight ",arguments.dict_path)
            self.model.load_state_dict(torch.load(arguments.dict_path, map_location=self.device))
        else:
            print("Did not load weight")
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.ent_pos = self.config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", None))
        print("entailment position:",self.ent_pos)
        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)
    
    def _run_batch(self, batch):
        # here
        with torch.no_grad():
           
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids)[0].detach().cpu().numpy()
            output = np.exp(output) / np.exp(output).sum(
                -1, keepdims=True
            ) 
            output0 = output[..., 0].reshape(input_ids.shape[0] // len(self.labels), -1)
            output1 = output[..., 1].reshape(input_ids.shape[0] // len(self.labels), -1)
            output2 = output[..., 2].reshape(input_ids.shape[0] // len(self.labels), -1)
            # for k, v0, v1, v2 in sorted(zip(batch, output0[0], output1[0],
            #                     output2[0]), key= lambda item : item[3] ,reverse= True)[0:6]:
            #     print("{}--{}, {}, {}".format(k, v0, v1, v2))


        return output2
    
    def __call__(
        self,
        features: List[EAEInputFeatures],
        batch_size: int = 1
    ):
        if not isinstance(features, list):
            features = [features]

        batch, outputs = [], []
        for i, feature in tqdm(enumerate(features), total=len(features)):
            sentences = [ 
                f"{feature.context} {self.tokenizer.sep_token} {label_template.format(arg=feature.argument,trg=feature.trigger, trg_subtype=feature.triggger_type.split('.')[-1])}."
                for label_template in self.labels
            ]
            batch.extend(sentences)

            if (i + 1) % batch_size == 0:
                output = self._run_batch(batch) # shape=[bs, prob(跑出来的p_entailment)] # [array([0.000954, 0.0...e=float16)]
                outputs.append(output)
                batch = []

        if len(batch) > 0:
            output = self._run_batch(batch)
            outputs.append(output)

        # print(outputs.__sizeof__())
        outputs = np.vstack(outputs)  # [n_data, prob(跑出来的p_entailment)]
        # print(outputs.shape)
        return outputs
    
class NLIEAEClassifierWithMappingHead(_NLIEAEClassifier):
    def __init__(self, 
                 *args,
                 labels: List[str],
                 template_mapping: Dict[str, str],
                 pretrained_model: str = arguments.model_path,
                 valid_conditions: Dict[str, List[str]],
                 **kwargs,
                 ):
        
        self.template_mapping_reverse = defaultdict(list) # key: template,  value: rel
        for key, value in template_mapping.items():
            for v in value:
                self.template_mapping_reverse[v].append(key)   
        self.new_topics = list(self.template_mapping_reverse.keys())  # 所有的template
        self.new_topics2id = {t: i for i, t in enumerate(self.new_topics)}  # template 2 id
        self.new_id2topics = dict(zip(self.new_topics2id.values(),self.new_topics2id.keys()))
        self.template_mapping2id = defaultdict(list)  # key:rel  value: template_id
        for key, value in template_mapping.items():
            self.template_mapping2id[key].extend([self.new_topics2id[v] for v in value])

        self.target_labels = labels  # 所有的label

        super().__init__(
            self.new_topics,
            *args,
            pretrained_model=pretrained_model,
            **kwargs,
        )


        self.valid_conditions = {}
        rel2id = {r: i for i, r in enumerate(labels)}
        self.n_rel = len(rel2id)
        for role, conditions in valid_conditions.items():
            for condition in conditions:
                if condition not in self.valid_conditions:
                    self.valid_conditions[condition] = np.zeros(self.n_rel)
                    self.valid_conditions[condition][rel2id["no_relation"]] = 1.0
                self.valid_conditions[condition][rel2id[role]] = 1.0
    

    def _apply_valid_conditions(self, probs, features: List[EAEInputFeatures]):
        mask_matrix = np.stack(
            [self.valid_conditions.get(feature.pair_type, np.zeros(self.n_rel)) for feature in features],  # 注意, wiki这里用ones
            axis=0,
        )
        probs = probs * mask_matrix

        return probs



    def __call__(self, features: List[EAEInputFeatures], batch_size=1):
       
        # if arguments.outputs==None or arguments.mode!="dev":
        #     print("outputs is None, compute out")
        #     outputs = super().__call__(features, batch_size)  # [n_data, entailment_prob(73个)]
        #     save_path = os.path.join(arguments.out_save_path,"{}_num{}_{}.pkl".format(
        #         arguments.dataset,len(outputs),TIME
        #         ))
        #     if arguments.mode!="dev":
        #         print("save to (initial): ",save_path)
        #         pickle_save(outputs,save_path) # 保存entailment的output
        # else: 
        #     print("load outputs")
        #     outputs = pickle_load(arguments.outputs)  # 用已经搞好了的mnli output


        outputs = super().__call__(features, batch_size)
        # outputs=> [n_sample, n_template]
        # 获取各个template的score,  从大到小排序
        outputs_sorted_index  = np.argsort(outputs,axis=1)[:,::-1] 
        template_sorted = list() # the same shape as outputs_sorted_index
        # 得到template
        for templates_out in outputs_sorted_index:
            template_sorted.append([self.new_id2topics[template_id] for template_id in templates_out])

        outputs = np.hstack(
            [
                np.max(outputs[:, self.template_mapping2id[label]], axis=-1, keepdims=True)
                if label in self.template_mapping2id
                else np.zeros((outputs.shape[0], 1))
                for label in self.target_labels
            ]
        )


        outputs = self._apply_valid_conditions(outputs, features=features)
        # outputs = np_softmax(outputs)

        return outputs, template_sorted


import json

if __name__ == "__main__":

    with open('/home/jwwang/URE_EAE/annotation/configs/ace.json', 'r') as file:
      config = json.load(file)

    labels = config['labels']
    template_mapping = config['template_mapping']
    valid_conditions = config['valid_conditions']

    
    clf = NLIEAEClassifierWithMappingHead(
        labels=labels,
        template_mapping=template_mapping,
        pretrained_model = "/data/transformers/microsoft_deberta-v2-xlarge-mnli",
        valid_conditions = valid_conditions
    )

    features = [
        EAEInputFeatures(
            context= "British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country 's energy regulator as the new chairman of finance watchdog the Financial Services Authority ( FSA ) .",
            argument= "Chancellor",
            role= "Person",
            trigger = "named",
            triggger_type= "Personnel.Nominate",
            entity_type= "PER",
            pair_type= "Personnel.Nominate:PER",
            text_piece=[],
            trigger_be="11",
            argument_be="aaa"
        )
    ]

    predictions = clf(features)

    print(predictions)

    for k, v in zip(predictions[0][0], labels):
        print('{}--{}'.format(k, v))
