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
import argparse



parser = argparse.ArgumentParser()
# 公共参数
parser.add_argument("--model_path", type=str,default="/data/transformers/microsoft_deberta-v2-xlarge-mnli", help="as named")
parser.add_argument("--cuda_index", type=int,default=3, help="as named")
parser.add_argument("--task_name", type=str,default="", help="")
parser.add_argument("--seed", type=int,default=16, help="as named")
parser.add_argument("--default_optimal_threshold", type=float,default=0.0, help="as named")
parser.add_argument("--generate_data", type=bool,default=False, help="是否保存annotated数据")
parser.add_argument("--generate_data_save_path", type=str,default="/home/jwwang/URE_share/outcome/annotation_result", help="保存annotated数据的文件夹")
parser.add_argument("--out_save_path", type=str,default="/home/jwwang/URE_share/outcome/annotation_result", help="保存annotated数据的文件夹")

# wiki

parser.add_argument("--dataset", type=str,default="wiki",choices=['tac','wiki'], help="as named")
parser.add_argument("--dict_path", type=str,default=None, help="fine-tuned model path")
parser.add_argument("--load_dict", type=bool,default=False, help="是否加载NLI的权重,fewshot的时候用")
parser.add_argument("--mode", type=str,default="test", help="annotate train set还是test set")
parser.add_argument("--run_evaluation_path", type=str,default="/data/tywang/URE_share/data/wiki/test.pkl", help="要annotate的数据路径")
parser.add_argument("--label2id_path", type=str,default="/data/tywang/URE_share/data/wiki/label2id.pkl", 
    help="自定义的label2id的路径(relation不一定和config中的index一样)"+
        "最终的label是这个label2id的label")
parser.add_argument("--config_path", type=str,default="/data/tywang/URE_share/annotation/configs/config_wiki_partial_constrain.json", help="")
parser.add_argument("--outputs", type=str,default=None, 
                help="要加载的以及跑好的entailment数据的路径")
args = parser.parse_args()


# 由外部决定的参数
dataset = args.dataset
run_evaluation_path = args.run_evaluation_path
config_path = args.config_path
default_optimal_threshold = args.default_optimal_threshold
label2id_path = args.label2id_path
generate_data = args.generate_data
generate_data_save_path = args.generate_data_save_path
model_path = args.model_path
mode = args.mode
task_name = args.task_name
outputs = args.outputs
cuda_index = args.cuda_index
load_dict = args.load_dict
dict_path = args.dict_path
seed = args.seed

# 推断的参数
get_optimal_threshold = True if dataset=="tac" else False
current_time = TIME
# out_save_path = os.path.join(CURR_DIR,"NLI_outputs/") # 保存entailment结果的地方
out_save_path=args.out_save_path
