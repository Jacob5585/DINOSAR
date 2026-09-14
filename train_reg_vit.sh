#!/bin/bash
python main_dino.py \
    --arch vit_small \
    --data_path datasets/imagenette2-160/train \
    --batch_size_per_gpu 40 \
    --epochs 100 \
    --use_lora True \
    --pretrained_path vit_small_patch16_224.pth
    # --pretrained_path dino_deitsmall16_pretrain.pth
    # --pretrained_path dino_deitsmall16_pretrain_full_checkpoint.pth