#!/bin/bash
torchrun --nproc_per_node=3 main_dino.py \
    --arch swin_tiny \
    --patch_size 4 \
    --window_size 7 \
    --data_path datasets/SARDet_100K/JPEGImages/val \
    --batch_size_per_gpu 10 \
    --epochs 100 \
    --in_chans 1 \
    --use_dense_prediction True \
    --use_lora True \
    --pretrained_path swin_tiny_patch4_window7_224.pth