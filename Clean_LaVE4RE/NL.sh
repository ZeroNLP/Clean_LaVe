SEED=""
CUDA_INDEX=""  
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET="" # choice in ["tac","wiki"]
n_rel=""
TRAIN_PATH="" # come from annotate result
specified_save_path=""


echo "NL begin !---------------------------------"
python -u ./NL/NL_noWeight.py \
                                    --seed ${SEED} \
                                    --epoch 10  \
                                    --cuda_index ${CUDA_INDEX} \
                                    --e_tags_path "./annotation/configs/${DATASET}_tags.pkl" \
                                    --n_rel ${n_rel} \
                                    --lr 4e-7 \
                                    --ln_neg ${n_rel} \
                                    --train_path ${TRAIN_PATH} \
                                    --specified_save_path ${specified_save_path} \
                                    --use_weight False