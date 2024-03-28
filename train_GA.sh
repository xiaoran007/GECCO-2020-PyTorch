#!/usr/bin/env bash

declare DATA_NAME="Credit"
declare TRAIN_DATA_PATH="./Datasets/${DATA_NAME}/train.csv"
declare SAMPLES_DIR="./Samples/${DATA_NAME}"

declare GA_CHECKPOINT_NAME="Test_1"
#declare GA_CHECKPOINT_PATH="./GA-Checkpoints/${DATA_NAME}/${GA_CHECKPOINT_NAME}/generation=100/checkpoint.pkl"
declare GA_CHECKPOINT_PATH="./GA-Checkpoints/generation=100/checkpoint.pkl"

declare MODEL_SAVE_PATH="./Trained-Classifiers/${DATA_NAME}/ga/${GA_CHECKPOINT_NAME}/classifier.pth"

declare CLASSIFIER_NUM_HIDDEN_LAYERS="2"
declare CLASSIFIER_LEARNING_RATE="0.0001"
declare CLASSIFIER_BETA_1="0.9"
declare CLASSIFIER_BETA_2="0.999"
declare CLASSIFIER_BATCH_SIZE="1024"
declare CLASSIFIER_NUM_EPOCHS="100"
#declare CLASSIFIER_RUN_DEVICE="cpu"
declare CLASSIFIER_RUN_DEVICE="cuda"

declare RAND_SEED="777"
declare VERBOSE="--verbose"

python3 train_GA.py \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --samples-dir "${SAMPLES_DIR}" \
    --ga-checkpoint-path "${GA_CHECKPOINT_PATH}" \
    --model-save-path "${MODEL_SAVE_PATH}" \
    --num-hidden-layers "${CLASSIFIER_NUM_HIDDEN_LAYERS}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --num-epochs "${CLASSIFIER_NUM_EPOCHS}" \
    --run-device "${CLASSIFIER_RUN_DEVICE}" \
    --learning-rate "${CLASSIFIER_LEARNING_RATE}" \
    --beta-1 "${CLASSIFIER_BETA_1}" \
    --beta-2 "${CLASSIFIER_BETA_2}" \
    --rand-seed "${RAND_SEED}" \
    "${VERBOSE}"
