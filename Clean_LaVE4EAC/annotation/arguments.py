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
parser.add_argument("--cuda_index", type=int,default=0, help="as named")
parser.add_argument("--seed", type=int,default=16, help="as named")
parser.add_argument("--default_optimal_threshold", type=float,default=0.0, help="as named")
parser.add_argument("--generate_data", type=bool,default=False, help="是否保存annotated数据")
parser.add_argument("--generate_data_save_path", type=str,default="/home/jwwang/URE_share/outcome/annotation_result", help="保存annotated数据的文件夹")

parser.add_argument("--output_path", type=str,default="", help="as named")
parser.add_argument("--threshold", type=float,default=-1.0, help="as named")


parser.add_argument("--dataset", type=str,default="wiki",choices=['ace'], help="as named")
parser.add_argument("--dict_path", type=str,default=None, help="fine-tuned model path")
parser.add_argument("--load_dict", type=bool,default=False, help="是否加载NLI的权重,fewshot的时候用")
parser.add_argument("--mode", type=str,default="test", help="annotate train set还是test set", choices=['test', 'train', '0.01dev', 'dev']) 
parser.add_argument("--run_evaluation_path", type=str,default="/data/tywang/URE_share/data/wiki/test.pkl", help="要annotate的数据路径")
parser.add_argument("--config_path", type=str,default="/data/tywang/URE_share/annotation/configs/config_wiki_partial_constrain.json", help="")

parser.add_argument("--out_save_path", type=str,default="", help="as named")
parser.add_argument("--outputs", type=str,default="", help="as named")

args = parser.parse_args()


# 由外部决定的参数
dataset = args.dataset
run_evaluation_path = args.run_evaluation_path
config_path = args.config_path
default_optimal_threshold = args.default_optimal_threshold
generate_data = args.generate_data
generate_data_save_path = args.generate_data_save_path
model_path = args.model_path
mode = args.mode
cuda_index = args.cuda_index
load_dict = args.load_dict
dict_path = args.dict_path
seed = args.seed
output_path = args.output_path
threshold = args.threshold
# 推断的参数
current_time = TIME
outputs= args.outputs
out_save_path=args.out_save_path
