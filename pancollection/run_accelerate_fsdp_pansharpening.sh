#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PATH=$(dirname $(which python))

export NCCL_TIMEOUT=600
export MASTER_PORT=24567
accelerate launch --config_file $SCRIPT_DIR/configs/accelerate_FSDP_config.yaml $PATH/accelerate_pansharpening \
--config-path=$SCRIPT_DIR/configs \
--config-name=model \
+plugins=["FSDP"] \
args.experimental_desc=fsdp \
# +args.import_path=[$WORK_DIR,models]
# args.dataset_type="Dummy" \
# max_epochs=1 \
# save_interval=1 \
# args.workflow='[["train", 1]]'
# gpu_ids=[0,1]





# single SAM: 3.32597, ERGAS: 2.46769, PSNR: 37.92744 Training time 4:02:32
# two machines SAM: 3.32597, ERGAS: 2.46769, PSNR: 37.92744 Training time 3:55:50

