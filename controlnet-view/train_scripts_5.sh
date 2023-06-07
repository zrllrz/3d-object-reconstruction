CUDA_VISIBLE_DEVICES=0 rm -rf ./models/control_sd21_view_ini.ckpt
CUDA_VISIBLE_DEVICES=0 python tool_add_control_sd21_view.py ./models/v2-1_512-ema-pruned.ckpt ./models/control_sd21_view_ini.ckpt
CUDA_VISIBLE_DEVICES=0 python train_view.py --checkdir='model_checkpoint_5' --split='train_5'
