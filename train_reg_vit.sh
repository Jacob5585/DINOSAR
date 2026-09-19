#!/bin/bash
python main_dino.py \
    --arch vit_small \
    --data_path datasets/SARDet_100K/JPEGImages/train_val \
    --batch_size_per_gpu 10 \
    --epochs 100 \
    --use_lora True \
    --pretrained_path deit_small_patch16_224.pth
    # --pretrained_path vit_small_patch16_224.pth
    # --pretrained_path dino_deitsmall16_pretrain.pth
    # --pretrained_path dino_deitsmall16_pretrain_full_checkpoint.pth