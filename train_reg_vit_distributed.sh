#!/bin/bash
torchrun --nproc_per_node=3 main_dino.py \
    --arch vit_small \
    --data_path datasets/SARDet_100K/JPEGImages/train_val \
    --batch_size_per_gpu 24 \
    --epochs 100 \
    --in_chans 1 \
    --use_lora True \
    --pretrained_path deit_small_patch16_224.pth