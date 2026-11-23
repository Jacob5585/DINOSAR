#!/bin/bash
python main_dino.py \
    --arch swin_tiny \
    --patch_size 4 \
    --window_size 7 \
    --data_path datasets/imagenette2-160/train \
    --batch_size_per_gpu 40 \
    --epochs 100 \
    --use_dense_prediction True \
    --use_lora True \
    --pretrained_path swin_tiny_esvit.pth
    # --use_lora True --pretrained_path swin_tiny_patch4_window7_224.pth