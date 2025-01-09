#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PATH=$(dirname $(which python))

# num2
export NCCL_TIMEOUT=600
export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=24567
accelerate launch --config_file $SCRIPT_DIR/configs/accelerate_config.yaml $PATH/accelerate_pansharpening \
--config-path=$SCRIPT_DIR/configs \
--config-name=model \
# +args.import_path=[$WORK_DIR,models]
# args.workflow='[["test", 1], ["train", 1]]'
# max_epochs=3
# args.dataset_type="Dummy"
# gpu_ids=[0,1]