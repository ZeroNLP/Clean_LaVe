TIME=$(date +"%Y-%m-%d-%H:%M:%S")

SEED=""
CUDA_INDEX=""  
MNLI_PATH="" # recommend microsoft_deberta-v2-xlarge-mnli
DATASET=""  
MODE=""
run_evaluation_path="" 

# 设置开始和结束值
start=0.01
end=0.3
step=0.02

# 初始化变量来记录当前值
current_value="$start"

pos_annotate_path="" # come from NL.sh result
confidence_path="" # come from annotate.sh result

# 使用循环遍历数字范围
while (( $(awk 'BEGIN {print ('"$current_value"' <= '"$end"')}') ))
do

    selected_data_path=""
    python ./NL/sampler.py \
                                --seed ${SEED} \
                                --dataset ${DATASET} \
                                --ratio ${current_value} \
                                --save_path "${selected_data_path}" \
                                --pos_annotate_path "${pos_annotate_path}" \
                                --confidence_path "${confidence_path}" \
                                --label2id "./annotation/configs/${DATASET}_label2id.pkl"


    echo "-----------------------------------------------------------------------"
    echo "当前值是：$current_value"
    # fintune
    TIME=$(date +"%Y-%m-%d-%H:%M:%S")
    check_point_path=""
    python -u ./finetune/fintune_mnli_v3.py \
                                --batch_size 6 \
                                --cuda_index ${CUDA_INDEX} \
                                --seed ${SEED} \
                                --lr 4e-7  \
                                --ratio ${current_value} \
                                --dataset ${DATASET} \
                                --model_path ${MNLI_PATH} \
                                --label2id_path "./annotation/configs/${DATASET}_label2id.pkl" \
                                --selected_data_path  ${selected_data_path} \
                                --config_path  "./annotation/configs/config_${DATASET}_partial_constrain.json" \
                                --template2label_path "./annotation/configs/${DATASET}_template2id.pkl" \
                                --epoch 10 \
                                --check_point_path ${check_point_path}



    # infer
    python -u ./annotation/run_evaluation.py \
                                --model_path ${MNLI_PATH} \
                                --cuda_index ${CUDA_INDEX} \
                                --dataset ${DATASET} \
                                --seed ${SEED} \
                                --mode "test" \
                                --run_evaluation_path "${run_evaluation_path}" \
                                --label2id_path "./annotation/configs/${DATASET}_label2id.pkl" \
                                --config_path "./annotation/configs/config_${DATASET}_partial_constrain.json" \
                                --load_dict True \
                                --generate_data_save_path '' \
                                --dict_path ${check_point_path} \
                                --out_save_path ""


    current_value=$(awk 'BEGIN {print ('"$current_value"' + '"$step"')}')
    rm "${check_point_path}"

    echo "-----------------------------------------------------------------------"
done
