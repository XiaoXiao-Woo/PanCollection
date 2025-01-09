#!/bin/bash
tmp=$(dirname "$(realpath "$0")")
SCRIPT_DIR=$(dirname "$tmp")


work_dir_path="/Data2/woo/results/pansharpening/test"


export NCCL_TIMEOUT=600
export CUDA_VISIBLE_DEVICES=1
export MASTER_PORT=24567
python $SCRIPT_DIR/python_scripts/run_naive_pansharpening.py \
--config-path=$SCRIPT_DIR/configs \
--config-name=model \
+work_dir=$work_dir_path \
# args.dataset_type="Dummy"
# gpu_ids=[0,1]
# mixed_precision=bf16 \


# export NCCL_TIMEOUT=600
# export CUDA_VISIBLE_DEVICES=1
# export MASTER_PORT=24567
# accelerate launch --config_file $SCRIPT_DIR/configs/accelerate_config.yaml $SCRIPT_DIR/python_scripts/accelerate_pansharpening.py \
# --config-path=$SCRIPT_DIR/configs \
# --config-name=model \
# +plugins=["FSDP"] \
# +work_dir=$work_dir_path
# # gpu_ids=[0,1]
# # mixed_precision=bf16 \

