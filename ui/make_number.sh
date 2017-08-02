#!/bin/bash
python main.py \
    --experiment-name mnist_16_simplecnn_7a_simpledeconv_a_mlp_2a_1501101651 \
    --encoder-input-filename ~/data/mnist_train.dataset \
    --decoder-input-filename ~/data/mnist_train.dataset \
    --encoder-datatype img \
    --decoder-datatype img \
    --encoder-model simplecnn_7a \
    --decoder-model simpledeconv_a \
    --discriminator-model simplecnn_7a \
    --classifier-model mlp_2a \
    --mode dream \
    --thought-vector-size 16 \
    --img-width-encoder 32 \
    --img-width-decoder 32 \
    --dream-examples 1 \
    --video-filename dream_out.mjpeg

ffmpeg -nostdin -y -i ~/results/mnist_16_simplecnn_7a_simpledeconv_a_mlp_2a_1501101651/dream_out.mjpeg number.mp4

rm ~/results/mnist_16_simplecnn_7a_simpledeconv_a_mlp_2a_1501101651/dream_out.mjpeg

mv number.mp4 ui/static/number-bar.mp4

