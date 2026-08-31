# Fine-grained visual categorization: linear probing on iNaturalist21, CUB-200-2011
# and Stanford-Cars, with exactly the same protocol as the ImageNet linear probe.
# We use torchrun to launch the job with 8 GPUs on a single node

GPU_NUM=8  # REPLACE with the number of GPUs you want to use
ADDR=localhost # REPLACE with the master node address if using multiple nodes
PORT=12345  # REPLACE with an available port number
DATASETS="inat21 cub200 cars"  # REPLACE with the subset of datasets you want to evaluate

for DATASET in $DATASETS; do
    CONFIG_FILE=./configs/classification/probing_${DATASET}.yaml
    echo "=== Linear probing on ${DATASET} (${CONFIG_FILE}) ==="
    torchrun \
        --nnodes=1 \
        --nproc-per-node=$GPU_NUM \
        --node_rank=0 \
        --master_addr=$ADDR \
        --master_port=$PORT \
        main.py --config_file $CONFIG_FILE --task classification
done
