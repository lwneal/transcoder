#!/bin/bash

function pass() {
    echo "[32mTest passed with exit code $?[0m"
}

function fail() {
    echo "[31mERROR: Test failed with exit code $?[0m"
    exit
}

# This command should work
python main.py \
 --encoder-input-filename ~/data/mnist_train.dataset \
 --encoder-datatype img \
 --decoder-input-filename ~/data/mnist_train.dataset \
 --decoder-datatype img \
 --thought-vector-size 2 \
 --encoder-weights mnist_twodim.h5 \
 --decoder-weights mnist_twodim.h5 \
 --discriminator-weights mnist_twodim.h5 \
 --pretrained-encoder vgg16 \
 --img-width 32 \
 --batches-per-epoch 10 \
 --epochs 1 \
 --enable-gan True \
 --mode train \
&& pass || fail


# This command doesn't work currently, it fails because of a bug
python main.py \
 --encoder-input-filename ~/data/cub200.dataset \
 --decoder-input-filename ~/data/cub200.dataset \
 --encoder-datatype img \
 --decoder-datatype lab \
 --encoder-weights cub200_classifier_encoder.h5 \
 --decoder-weights cub200_classifier_decoder.h5 \
 --discriminator-weights /dev/null \
 --enable-gan False \
 --img-width 128 \
 --batches-per-epoch 100 \
 --thought-vector-size 512 \
 --mode test \
&& pass || fail

