#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=24567
python $SCRIPT_DIR/python_scripts/run_naive_pansharpening.py \
--config-path=$SCRIPT_DIR/configs \
--config-name=model \
# +work_dir=/Data2/woo/results/test/pansharpening
# +args.import_path=[$WORK_DIR,models]