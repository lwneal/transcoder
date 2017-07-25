#!/bin/bash
set -e

DATASET=$1
ENCODER=$2
DECODER=$3
CLASSIFIER=$4
THOUGHT_VECTOR_SIZE=$5
EPOCHS=$6
P_LAYERS=$7
IMG_WIDTH=$8
TIMESTAMP=$9
EVALUATE_MODE=$10

if [[ $# -lt 7 ]]; then
    echo "Usage: $0 <dataset> <encoder> <decoder> <classifier> <latent_size> <epochs> <p_layers> <img_width> [timestamp] [evaluate_mode]"
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
 --img-width-encoder $IMG_WIDTH \
 --img-width-decoder $IMG_WIDTH \
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
 --img-width-encoder $IMG_WIDTH \
 --img-width-decoder $IMG_WIDTH \
 --enable-gan False \
 --enable-classifier False \
 --stdout-filename evaluate_classifier.txt \
 --mode evaluate

# Generate a "dream" video
TMP_ID=`date +%s`
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
 --img-width-encoder $IMG_WIDTH \
 --img-width-decoder $IMG_WIDTH \
 --enable-gan False \
 --stdout-filename dream.txt \
 --video-filename dream_output_${TMP_ID}.mjpeg \
 --mode dream

# Re-encode the video to mp4 for storage
pushd ~/results/$EXPERIMENT_NAME
ffmpeg -y -i dream_output_${TMP_ID}.mjpeg dream_${TMP_ID}.mp4
rm dream_output_${TMP_ID}.mjpeg
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
     --img-width-encoder $IMG_WIDTH \
     --img-width-decoder $IMG_WIDTH \
     --enable-gan False \
     --enable-classifier True \
     --stdout-filename counterfactual.txt \
     --video-filename counterfactual_output_${TMP_ID}.mjpeg \
     --mode counterfactual
done

# Re-encode the video to mp4 for storage
pushd ~/results/$EXPERIMENT_NAME
ffmpeg -y -i counterfactual_output_${TMP_ID}.mjpeg counterfactuals_${TMP_ID}.mp4
rm counterfactual_output_${TMP_ID}.mjpeg 
popd

touch ~/results/$EXPERIMENT_NAME/finished
