TIME=$(date +"%Y-%m-%d-%H:%M:%S")


MNLI_PATH="" # recommend mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
CUDA_INDEX=1
DATASET="smiler"
SEED="17"




MODE="test"
run_evaluation_path="" 


start=0.001
end=0.01
step=0.001

current_value="$start"

pos_annotate_path=""
confidence_path=""

while (( $(awk 'BEGIN {print ('"$current_value"' <= '"$end"')}') ))
do

    echo "-----------------------------------------------------------------------"
    echo "current_value: $current_value"
    TIME=$(date +"%Y-%m-%d-%H:%M:%S")
    check_point_path=""
    python -u ./finetune/fintune_mnli_v3.py \
                                --ratio ${current_value} \
                                --config_path  "" \
                                --check_point_path ${check_point_path} \
                                --sortByPlabelConfidence_path "${confidence_path}" \
                                --train_path "${pos_annotate_path}"



    # infer
    python -u ./annotation/run_evaluation.py \
                                --model_path ${MNLI_PATH} \
                                --cuda_index ${CUDA_INDEX} \
                                --dataset ${DATASET} \
                                --seed ${SEED} \
                                --mode "test" \
                                --run_evaluation_path "${run_evaluation_path}" \
                                --label2id_path "" \
                                --config_path "" \
                                --load_dict True \
                                --generate_data_save_path '' \
                                --dict_path ${check_point_path} \
                                --out_save_path ""


    current_value=$(awk 'BEGIN {print ('"$current_value"' + '"$step"')}')
    rm "${check_point_path}"

    echo "-----------------------------------------------------------------------"
done
