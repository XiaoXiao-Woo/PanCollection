#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")


# export CUDA_VISIBLE_DEVICES=1 
# export MASTER_PORT=24567
# udl_cil --config-path=/Data2/woo/NIPS/PanCollection/pancollection/configs \
# --config-name=fusionnet.yaml



export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=24567
python $SCRIPT_DIR/python_scripts/run_mmcv_pansharpening.py \
--config-path=$SCRIPT_DIR/configs \
--config-name=model
# +args.import_path=[$WORK_DIR,models]