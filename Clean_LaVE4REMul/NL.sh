
PROJECT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAIN_PATH=""
specified_save_path=""
DEV_PATH=""
CONFIG_PATH=""


echo "NL begin !---------------------------------"
python -u ./NL/NL.py \
        --e_tags_path "" \
        --train_path ${TRAIN_PATH} \
        --dev_path ${DEV_PATH} \
        --config_path ${CONFIG_PATH} \
        --specified_save_path ${specified_save_path}