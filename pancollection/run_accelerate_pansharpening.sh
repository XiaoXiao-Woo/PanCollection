#!/bin/bash
SCRIPT_DIR=$(python -c "from pancollection import pancollection_dir; print(pancollection_dir)")
PYTHON_PATH=$(dirname $(which python))

MIN_PORT=1024
MAX_PORT=65535

is_port_in_use() {
    local port=$1
    if netstat -tuln | grep -q ":$port "; then
        return 0 
    else
        return 1
    fi
}


generate_random_port() {
    local port
    while true; do
        port=$((RANDOM % (MAX_PORT - MIN_PORT + 1) + MIN_PORT))
        if ! is_port_in_use $port; then
            echo $port
            return
        fi
    done
}

multi_gpu() {
    if [ "$1" -gt 1 ]; then
        echo "--multi_gpu --num_processes=$1"
    else
        echo ""
    fi
}

available_port=$(generate_random_port)

# export NCCL_TIMEOUT=600
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --master_port $available_port --config_file $SCRIPT_DIR/configs/accelerate_config.yaml $PATH/accelerate_pansharpening \
# --config-path=$SCRIPT_DIR/configs \
# --config-name=model \
# # +args.import_path=[$WORK_DIR,models]
# # args.workflow='[["test", 1], ["train", 1]]'
# # max_epochs=3
# # args.dataset_type="Dummy"
# # gpu_ids=[0,1]


udl_cil \
--config-path=/home/dsq/nips/PanCollection/pancollection/configs \
--config-name=fusionnet \
+command=["accelerate","launch","--multi_gpu",'"--num_processes=2"',"--main_process_port","$available_port","--config_file","$SCRIPT_DIR/configs/accelerate_config.yaml"] \
+run_file=$PYTHON_PATH/accelerate_pansharpening \
+sampler="GridSampler" \
+n_jobs=1 \
experimental_desc=test \
+args.import_path="['$WORK_DIR','models']" \
+foreground=true \
+max_retry=2 \
# +gpu_ids=[6,7]
# +skip_rerun=false \