# ADE20K semantic segmentation (UPerNet, 160k iterations, global batch 16)
# We use torchrun to launch the job with 8 GPUs on a single node
# By default, it support 8 GPUs; change --nproc-per-node if needed

GPU_NUM=8  # REPLACE with the number of GPUs you want to use
ADDR=localhost # REPLACE with the master node address if using multiple nodes
PORT=12345  # REPLACE with an available port number
CONFIG_FILE=./configs/segmentation/ade20k_upernet_vitb16.yaml  # REPLACE with your config file path

torchrun \
    --nnodes=1 \
    --nproc-per-node=$GPU_NUM \
    --node_rank=0 \
    --master_addr=$ADDR \
    --master_port=$PORT \
    main.py --config_file $CONFIG_FILE --task segmentation
