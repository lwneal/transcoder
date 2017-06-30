#!/bin/bash
set -e

ENCODER_MODEL=$1
DECODER_MODEL=$2
CLASSIFIER_MODEL=$3
THOUGHT_VECTOR_SIZE=$3
EXPERIMENT_NAME=mnist_${THOUGHT_VECTOR_SIZE}_${ENCODER_MODEL}_${DECODER_MODEL}_`date +%s`

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <encoder_model> <decoder_model> <thought_vector_size>"
    exit
fi

scripts/download_mnist.py

# First train a manifold
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/mnist_train.dataset \
 --decoder-input-filename ~/data/mnist_train.dataset \
 --classifier-input-filename ~/data/mnist_train.dataset \
 --encoder-datatype img \
 --decoder-datatype img \
 --encoder-model $ENCODER_MODEL \
 --decoder-model $DECODER_MODEL \
 --classifier-model $CLASSIFIER_MODEL \
 --discriminator-model $ENCODER_MODEL \
 --thought-vector-size $THOUGHT_VECTOR_SIZE \
 --epochs 10 \
 --batches-per-epoch 200 \
 --stdout-filename train_aae.txt \
 --mode train

# Evaluate the classifier
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/mnist_test.dataset \
 --decoder-input-filename ~/data/mnist_test.dataset \
 --vocabulary-filename ~/data/mnist_train.dataset \
 --encoder-datatype img \
 --decoder-datatype lab \
 --encoder-model $ENCODER_MODEL \
 --decoder-model $CLASSIFIER_MODEL \
 --thought-vector-size $THOUGHT_VECTOR_SIZE \
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
 --encoder-model $ENCODER_MODEL \
 --decoder-model $DECODER_MODEL \
 --thought-vector-size $THOUGHT_VECTOR_SIZE \
 --enable-gan False \
 --stdout-filename dream.txt \
 --video-filename dream_output.mjpeg \
 --mode dream

# Re-encode the video to mp4 for storage
cd ~/results/$EXPERIMENT_NAME
ffmpeg -i dream_output.mjpeg dream.mp4
rm dream_output.mjpeg

# Add counterfactuals
for i in `seq 10`; do 
    python main.py \
     --experiment-name $EXPERIMENT_NAME \
     --encoder-input-filename ~/data/mnist_test.dataset \
     --decoder-input-filename ~/data/mnist_test.dataset \
     --vocabulary-filename ~/data/mnist_train.dataset \
     --encoder-datatype img \
     --decoder-datatype lab \
     --encoder-model $ENCODER_MODEL \
     --decoder-model $DECODER_MODEL \
     --thought-vector-size $THOUGHT_VECTOR_SIZE \
     --enable-gan False \
     --stdout-filename counterfactual.txt \
     --mode counterfactual
done
