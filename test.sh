#!/bin/bash

function pass() {
    echo "[32mTest passed with exit code $?[0m"
}

function fail() {
    echo "[31mERROR: Test failed with exit code $?[0m"
    exit 1
}

function cleanup() {
    # NOTE: Don't start an experiment name with "unittest"
    rm -rf ~/results/unittest*
    echo Cleaning up unit test output...
}

function test1() {
    # Test that we can train an autoencoder with a GAN
    python main.py \
     --experiment-name unittest123 \
     --encoder-input-filename ~/data/mnist_train.dataset \
     --encoder-datatype img \
     --decoder-input-filename ~/data/mnist_train.dataset \
     --decoder-datatype img \
     --thought-vector-size 2 \
     --encoder-weights mnist_manifold_2dim_enc.h5 \
     --decoder-weights mnist_manifold_2dim_dec \
     --discriminator-weights mnist_manifold_2dim_disc.h5 \
     --pretrained-encoder vgg16 \
     --img-width 32 \
     --batches-per-epoch 10 \
     --batch-size 1 \
     --epochs 1 \
     --enable-gan True \
     --mode train
}


function test2() {
    # Test that we can evaluate on a dataset with odd length
    python main.py \
     --experiment-name unittest123 \
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
     --batch-size 16 \
     --thought-vector-size 512 \
     --mode test
}

cleanup

for i in `seq 2`; do
    test$i && pass || fail
    cleanup
done
