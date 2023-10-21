

SEED="17"
CUDA_INDEX="3"  
MNLI_PATH="" # recommend microsoft_deberta-v2-xlarge-mnli
DATASET="ace"  
MODE="test"
run_evaluation_path="" 

# 设置开始和结束值
start=0.01
end=0.1
step=0.01

# 初始化变量来记录当前值
current_value="$start"

sortByPlabelConfidence_path=""
train_path=""

# 使用循环遍历数字范围
while (( $(awk 'BEGIN {print ('"$current_value"' <= '"$end"')}') ))
do
    echo "-----------------------------------------------------------------------"
    echo "当前值是：$current_value"
    # fintune
    TIME=$(date +"%Y-%m-%d-%H:%M:%S")
    check_point_path=""
    python -u ./finetune/fintune_mnli_v3.py  \
                                    --sortByPlabelConfidence_path "${sortByPlabelConfidence_path}" \
                                    --train_path "${train_path}" \
                                    --ratio "$current_value" \
                                    --check_point_path "${check_point_path}" \
                                    --cuda_index ${CUDA_INDEX} \
                                    --config_path ""



    # infer
    python -u ./annotation/run_evaluation.py \
                                    --model_path ${MNLI_PATH} \
                                    --cuda_index ${CUDA_INDEX} \
                                    --dataset ${DATASET} \
                                    --seed ${SEED} \
                                    --mode ${MODE} \
                                    --run_evaluation_path ${run_evaluation_path} \
                                    --config_path "" \
                                    --generate_data False \
                                    --generate_data_save_path '' \
                                    --load_dict True \
                                    --dict_path "${check_point_path}" \
                                    --threshold 0


    current_value=$(awk 'BEGIN {print ('"$current_value"' + '"$step"')}')
    echo "-----------------------------------------------------------------------"
done
