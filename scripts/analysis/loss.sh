# For visualization, we use single GPU
export CUDA_VISIBLE_DEVICES=0  # Set to the GPU ID you want to use
CONFIG_FILE=./configs/analysis/pred_loss.yaml  # REPLACE with your config file path
python main.py --config_file $CONFIG_FILE --task analysis