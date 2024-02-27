#!/bin/bash

DIR="CKPTS/VIT-LORA-3CKPTS/" 
DATASET="CIFAR10"
DATA_PATH="data/cifar10/"
TRANSFORM="VGG"
MODEL="LORA_BNN"
WD=0.0005
CKPT=("CKPTS/VIT-LORA-3CKPTS/best_model_0_noise_std_0.pt","CKPTS/VIT-LORA-3CKPTS/best_model_1_noise_std_0.pt","CKPTS/VIT-LORA-3CKPTS/best_model_2_noise_std_0.pt")

# Run the Python script
CUDA_VISIBLE_DEVICES=1 python3 surface_finder.py --dir="$DIR" \
                 --dataset="$DATASET" \
                 --data_path="$DATA_PATH" \
                 --transform="$TRANSFORM" \
                 --model="$MODEL" \
                 --wd="$WD" \
                 --use_test \
                 --ckpt="$CKPT"