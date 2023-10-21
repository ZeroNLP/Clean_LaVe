PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED="16"
CUDA_INDEX="0"  
MNLI_PATH="" #recommend mDeBERTa-v3-base-xnli-multilingual-nli-2mil7

DATASET="smiler"  # choice in ["tac","wiki"]          
MODE="dev"
run_evaluation_path=""



echo $run_evaluation_path
python -u ./annotation/run_evaluation.py \
                                --model_path ${MNLI_PATH} \
                                --cuda_index ${CUDA_INDEX} \
                                --dataset ${DATASET} \
                                --seed ${SEED} \
                                --mode ${MODE} \
                                --run_evaluation_path ${run_evaluation_path} \
                                --label2id_path "" \
                                --config_path "" \
                                --generate_data True \
                                --generate_data_save_path '' \
                                --out_save_path '' \
