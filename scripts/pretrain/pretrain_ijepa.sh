# We use torchrun to launch the training with 8 GPUs on a single node
# By default, it support 8 GPUs; change --nproc-per-node if needed

GPU_NUM=8  # REPLACE with the number of GPUs you want to use
ADDR=localhost # REPLACE with the master node address if using multiple nodes
PORT=12345  # REPLACE with an available port number
CONFIG_FILE=./configs/ijepa/vitt16_in1k.yaml  # REPLACE with your config file path

torchrun \
    --nnodes=1 \
    --nproc-per-node=$GPU_NUM \
    --node_rank=0 \
    --master_addr=$ADDR \
    --master_port=$PORT \
    main.py --config_file $CONFIG_FILE --task pretrain