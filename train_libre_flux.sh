#!/bin/bash

# This script runs the LibreFLUX training process

/venv/main/bin/python train_libre_flux.py \
  --pretrained_model_name_or_path="jimmycarter/LibreFLUX" \
  --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --data_json_file="laion2b-squareish-1024px/data.json" \
  --data_root_path="laion2b-squareish-1024px" \
  --val_data_json_file="test_dataset/data.json" \
  --val_data_root_path="test_dataset" \
  --mixed_precision="bf16" \
  --resolution=512 \
  --train_batch_size=6 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --quantize \
  --output_dir="output_dir" \
  --save_steps=1000 \
  --val_steps=1000 \
  --pretrained_ip_adapter_path="output_dir/checkpoint-0058000.pt"

echo "Training script finished."