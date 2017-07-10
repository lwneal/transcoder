#!/bin/bash
set -e

DATASET=$1
ENCODER=$2
DECODER=$3
CLASSIFIER=$4
THOUGHT_VECTOR_SIZE=$5
EPOCHS=$6
P_LAYERS=$7
TIMESTAMP=$8
EVALUATE_MODE=$9

if [[ $# -lt 7 ]]; then
    echo "Usage: $0 <dataset> <encoder> <decoder> <classifier> <latent_size> <epochs> <p_layers> [timestamp] [evaluate_mode]"
    exit
fi

if [[ -z $TIMESTAMP ]]; then
    TIMESTAMP=`date +%s`
    echo "No experiment ID provided, starting new experiment $TIMESTAMP"
fi

EXPERIMENT_NAME=${DATASET}_${THOUGHT_VECTOR_SIZE}_${ENCODER}_${DECODER}_${CLASSIFIER}_${TIMESTAMP}

scripts/download_${DATASET}.py

set -x
# First train a manifold
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/${DATASET}_train.dataset \
 --decoder-input-filename ~/data/${DATASET}_train.dataset \
 --classifier-input-filename ~/data/${DATASET}_train.dataset \
 --encoder-datatype img \
 --decoder-datatype img \
 --classifier-datatype lab \
 --encoder-model $ENCODER \
 --decoder-model $DECODER \
 --classifier-model $CLASSIFIER \
 --discriminator-model $ENCODER \
 --thought-vector-size $THOUGHT_VECTOR_SIZE \
 --epochs $EPOCHS \
 --perceptual-loss-layers $P_LAYERS \
 --batches-per-epoch 200 \
 --stdout-filename train_aae.txt \
 --enable-classifier True \
 --mode train


# Evaluate the classifier
# Uses classifier.h5 as the decoder, then disables the classifier
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/${DATASET}_test.dataset \
 --decoder-input-filename ~/data/${DATASET}_test.dataset \
 --vocabulary-filename ~/data/${DATASET}_train.dataset \
 --encoder-datatype img \
 --decoder-datatype lab \
 --decoder-weights classifier_${CLASSIFIER}.h5 \
 --encoder-model $ENCODER \
 --decoder-model $CLASSIFIER \
 --thought-vector-size $THOUGHT_VECTOR_SIZE \
 --enable-gan False \
 --enable-classifier False \
 --stdout-filename evaluate_classifier.txt \
 --mode evaluate

# Generate a "dream" video
python main.py \
 --experiment-name $EXPERIMENT_NAME \
 --encoder-input-filename ~/data/${DATASET}_test.dataset \
 --decoder-input-filename ~/data/${DATASET}_test.dataset \
 --vocabulary-filename ~/data/${DATASET}_train.dataset \
 --encoder-datatype img \
 --decoder-datatype img \
 --encoder-model $ENCODER \
 --decoder-model $DECODER \
 --thought-vector-size $THOUGHT_VECTOR_SIZE \
 --enable-gan False \
 --stdout-filename dream.txt \
 --video-filename dream_output.mjpeg \
 --mode dream

# Re-encode the video to mp4 for storage
pushd ~/results/$EXPERIMENT_NAME
ffmpeg -i dream_output.mjpeg dream.mp4
rm dream_output.mjpeg
popd

# Add counterfactuals
for i in `seq 10`; do 
    python main.py \
     --experiment-name $EXPERIMENT_NAME \
     --encoder-input-filename ~/data/${DATASET}_test.dataset \
     --decoder-input-filename ~/data/${DATASET}_test.dataset \
     --classifier-input-filename ~/data/${DATASET}_test.dataset \
     --vocabulary-filename ~/data/${DATASET}_train.dataset \
     --encoder-datatype img \
     --decoder-datatype img \
     --classifier-datatype lab \
     --encoder-model $ENCODER \
     --decoder-model $DECODER \
     --classifier-model $CLASSIFIER \
     --thought-vector-size $THOUGHT_VECTOR_SIZE \
     --enable-gan False \
     --enable-classifier True \
     --stdout-filename counterfactual.txt \
     --video-filename counterfactual_output.mjpeg \
     --mode counterfactual
done

# Re-encode the video to mp4 for storage
pushd ~/results/$EXPERIMENT_NAME
ffmpeg -i counterfactual_output.mjpeg counterfactuals.mp4
rm counterfactual_output.mjpeg 
popd

touch ~/results/$EXPERIMENT_NAME/finished
