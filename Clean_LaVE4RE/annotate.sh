PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SEED=""
CUDA_INDEX=""  
MNLI_PATH="" # recommend microsoft_deberta-v2-xlarge-mnli

DATASET=""  # choice in ["tac","wiki"]       
MODE="" # choice in ["test","dev","train"]
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
                                --config_path "./annotation/configs/config_${DATASET}_partial_constrain.json" \
                                --generate_data True \
                                --generate_data_save_path '' \
                                --out_save_path '' \