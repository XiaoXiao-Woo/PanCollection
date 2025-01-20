#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PATH=$(dirname $(which python))

work_dir_path="/Data3/woo/results"



export NCCL_TIMEOUT=600
export MASTER_PORT=24567
accelerate launch --use_deepspeed --config_file $SCRIPT_DIR/configs/accelerate_DS_config.yaml $PATH/accelerate_pansharpening \
--config-path=$SCRIPT_DIR/configs \
--config-name=model \
+plugins=["DeepSpeed"] \
+work_dir=$work_dir_path \
# +args.import_path=[$WORK_DIR,models]
# args.dataset_type="Dummy"
# gpu_ids=[0,1]

