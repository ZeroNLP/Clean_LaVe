import json
import pickle
import os
from pathlib import Path
import sys

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
print(CURR_DIR) # 加载模块；路径不适用

config_path = "/home/jwwang/URE_EAE/NL/config/ace.json"
label2id_path = "/home/jwwang/URE_EAE/NL/config/label2id.pkl"

label2id = {}
with open(config_path, 'r') as file:
    data = json.load(file)
    print(data['labels'])
    for index, label in enumerate(data['labels']):
        label2id[label] = index
    
    with open(label2id_path, 'wb') as fileout:
        pickle.dump(label2id, fileout)
    