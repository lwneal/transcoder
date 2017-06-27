#!/bin/bash
python main.py  --encoder-input-filename ~/data/cub200_train.dataset  --decoder-input-filename ~/data/cub200_train.dataset  --encoder-datatype img  --decoder-datatype img  --encoder-weights cub200_auto_enc.h5  --decoder-weights cub200_auto_dec.h5  --discriminator-weights cub200_auto_disc.h5  --pretrained-encoder vgg16  --epochs 9999  --enable-gan True  --img-width 128  --batches-per-epoch 100  --thought-vector-size 512  --mode train

