#!/bin/bash

# Download the dataset and generate inpainting examples
scripts/download_cub200.py
scripts/cub200_inpainting.py

# Run the experiment
python main.py \
    --encoder-input-filename ~/data/cub200_damaged.dataset \
    --encoder-datatype img \
    --decoder-input-filename ~/data/cub200.dataset \
    --decoder-datatype img \
    --encoder-weights inpainting_enc.h5 \
    --decoder-weights inpainting_dec.h5 \
    --discriminator-weights inpainting_disc.h5 \
    --thought-vector-size 64 \
    --img-width 256 \
    --csr-layers 1 \
    --enable-gan True \
    --batches-per-epoch 100 \
    --epochs 100
