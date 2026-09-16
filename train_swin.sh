#!/bin/bash
python main_dino.py \
    --arch swin_tiny \
    --patch_size 4 \
    --window_size 7 \
    --data_path datasets/SARDet_100K/JPEGImages/val \
    --batch_size_per_gpu 2 \
    --epochs 100 \
    --use_dense_prediction True \
    --use_lora True \
    --pretrained_path swin_tiny_esvit.pth