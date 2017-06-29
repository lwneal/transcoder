#!/bin/bash
set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 encoder_model decoder_model"
    exit
fi
ENCODER=$1
DECODER=$2

EXPERIMENT_NAME=cub200_big_128dim_${ENCODER}_${DECODER}_`date +%s`

scripts/download_cub200.py

# First train a manifold
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/cub200_train.dataset \
 --decoder-input-filename ~/data/cub200_train.dataset \
 --encoder-datatype img \
 --decoder-datatype img \
 --encoder-model $ENCODER \
 --decoder-model $DECODER \
 --discriminator-model $ENCODER \
 --thought-vector-size 128 \
 --epochs 100 \
 --batches-per-epoch 400 \
 --img-width 128 \
 --stdout-filename train_aae.txt \
 --mode train

# Now train a classifier keeping the encoder fixed
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/cub200_train.dataset \
 --decoder-input-filename ~/data/cub200_train.dataset \
 --encoder-datatype img \
 --decoder-datatype lab \
 --encoder-model $ENCODER \
 --decoder-model label_decoder \
 --thought-vector-size 128 \
 --freeze-encoder True \
 --enable-gan False \
 --epochs 10 \
 --batches-per-epoch 200 \
 --img-width 128 \
 --stdout-filename train_classifier.txt \
 --mode train

# Now evaluate the classifier
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/cub200_test.dataset \
 --decoder-input-filename ~/data/cub200_test.dataset \
 --vocabulary-filename ~/data/cub200_train.dataset \
 --encoder-datatype img \
 --decoder-datatype lab \
 --encoder-model $ENCODER \
 --decoder-model label_decoder \
 --thought-vector-size 128 \
 --enable-gan False \
 --img-width 128 \
 --stdout-filename evaluate_classifier.txt \
 --mode evaluate

# Generate a "dream" video
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/cub200_test.dataset \
 --decoder-input-filename ~/data/cub200_test.dataset \
 --vocabulary-filename ~/data/cub200_train.dataset \
 --encoder-datatype img \
 --decoder-datatype img \
 --encoder-model $ENCODER \
 --decoder-model $DECODER \
 --thought-vector-size 64 \
 --img-width 128 \
 --enable-gan False \
 --stdout-filename dream.txt \
 --video-filename dream_output.mjpeg \
 --mode dream


# Re-encode the video to mp4 for storage
cd ~/results/$EXPERIMENT_NAME
ffmpeg -nostdin -i dream_output.mjpeg dream.mp4
rm dream_output.mjpeg


# Perform counterfactual
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/cub200_test.dataset \
 --decoder-input-filename ~/data/cub200_test.dataset \
 --classifier-input-filename ~/data/cub200_test.dataset \
 --vocabulary-filename ~/data/cub200_train.dataset \
 --encoder-datatype img \
 --decoder-datatype img \
 --classifier-datatype lab \
 --encoder-model $ENCODER \
 --decoder-model $DECODER \
 --discriminator-model $ENCODER \
 --classifier-model label_decoder \
 --img-width 128 \
 --thought-vector-size 128 \
 --enable-classifier True \
 --mode counterfactual

