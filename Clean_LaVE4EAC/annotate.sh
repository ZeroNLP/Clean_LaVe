PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED="16"
CUDA_INDEX="2"  
MNLI_PATH="/data/transformers/microsoft_deberta-v2-xlarge-mnli"
# MNLI_PATH="/data/transformers/roberta-large-mnli/"

DATASET="ace"  
MODE="test"        
run_evaluation_path="/data/jwwang/URE_EAE/ace_dy/Dnorelation/"${MODE}".json" 

#--generate_data 表示产生数据
python -u ./annotation/run_evaluation.py \
                                --model_path ${MNLI_PATH} \
                                --cuda_index ${CUDA_INDEX} \
                                --dataset ${DATASET} \
                                --seed ${SEED} \
                                --mode ${MODE} \
                                --run_evaluation_path ${run_evaluation_path} \
                                --config_path "/home/jwwang/URE_EAE/annotation/configs/config_ace_constrain.json" \
                                --generate_data False \
                                --generate_data_save_path '/data/jwwang/URE_UEE/output/annotate_data' \
                                --threshold '0'
                                # --output_path '/data/jwwang/URE_UEE/output/annotate_data/time09-27-17*53*09_output.pkl' \
                                # --threshold '0.9'
