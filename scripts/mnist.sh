#!/bin/bash

EXPERIMENT_NAME=mnist_32dim_`date +%s`

scripts/download_mnist.py

# First train a manifold
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/mnist_train.dataset \
 --decoder-input-filename ~/data/mnist_train.dataset \
 --encoder-datatype img \
 --decoder-datatype img \
 --encoder-model simplecnn_7a \
 --decoder-model simpledeconv_a \
 --discriminator-model simplecnn_7a \
 --thought-vector-size 32 \
 --epochs 10 \
 --batches-per-epoch 200 \
 --stdout-filename train_aae.txt \
 --mode train

# Now train a classifier keeping the encoder fixed
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/mnist_train.dataset \
 --decoder-input-filename ~/data/mnist_train.dataset \
 --encoder-datatype img \
 --decoder-datatype lab \
 --encoder-model simplecnn_7a \
 --decoder-model label_decoder \
 --thought-vector-size 32 \
 --freeze-encoder True \
 --enable-gan False \
 --epochs 10 \
 --batches-per-epoch 200 \
 --stdout-filename train_classifier.txt \
 --mode train

# Now evaluate the classifier
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/mnist_test.dataset \
 --decoder-input-filename ~/data/mnist_test.dataset \
 --vocabulary-filename ~/data/mnist_train.dataset \
 --encoder-datatype img \
 --decoder-datatype lab \
 --encoder-model simplecnn_7a \
 --decoder-model label_decoder \
 --thought-vector-size 32 \
 --enable-gan False \
 --stdout-filename evaluate_classifier.txt \
 --mode evaluate

# Generate a "dream" video
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/mnist_test.dataset \
 --decoder-input-filename ~/data/mnist_test.dataset \
 --vocabulary-filename ~/data/mnist_train.dataset \
 --encoder-datatype img \
 --decoder-datatype img \
 --encoder-model simplecnn_7a \
 --decoder-model simpledeconv_a \
 --thought-vector-size 32 \
 --enable-gan False \
 --stdout-filename dream.txt \
 --video-filename dream_output.mjpeg \
 --mode dream

# Re-encode the video to mp4 for storage
ffmpeg -i dream_output.mjpeg dream.mp4
rm dream_output.mjpeg
