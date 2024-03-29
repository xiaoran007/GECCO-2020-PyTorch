#!/usr/bin/env bash

declare DATA_NAME="Credit"
declare TRAIN_DATA_PATH="./Datasets/${DATA_NAME}/train.csv"
declare SAMPLES_DIR="./Samples"

declare CLASSIFIER_NUM_HIDDEN_LAYERS="2"
declare CLASSIFIER_LEARNING_RATE="0.0001"
declare CLASSIFIER_BETA_1="0.9"
declare CLASSIFIER_BETA_2="0.999"
declare CLASSIFIER_BATCH_SIZE="1024"
declare CLASSIFIER_NUM_EPOCHS="100"
declare CLASSIFIER_RUN_DEVICE="cpu"
#declare CLASSIFIER_RUN_DEVICE="cuda"

declare RAND_SEED="777"
declare VERBOSE="--verbose"

python3 search_GA.py \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --samples-dir "${SAMPLES_DIR}" \
    --ratio-min 1.0 \
    --ratio-max 5.0 \
    --ga-population-size 100 \
    --ga-selection-method "roulette" \
    --ga-crossover-method "onepoint" \
    --ga-crossover-size 2 \
    --ga-mutation-method "swap" \
    --ga-mutation-rate 0.001 \
    --ga-replacement-method "parents" \
    --ga-num-generations 100 \
    --ga-checkpoint-dir "./GA-Checkpoints" \
    --classifier-num-hidden-layers "${CLASSIFIER_NUM_HIDDEN_LAYERS}" \
    --classifier-batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --classifier-num-epochs "${CLASSIFIER_NUM_EPOCHS}" \
    --classifier-run-device "${CLASSIFIER_RUN_DEVICE}" \
    --classifier-learning-rate "${CLASSIFIER_LEARNING_RATE}" \
    --classifier-beta-1 "${CLASSIFIER_BETA_1}" \
    --classifier-beta-2 "${CLASSIFIER_BETA_2}" \
    --rand-seed "${RAND_SEED}" \
    "${VERBOSE}"
