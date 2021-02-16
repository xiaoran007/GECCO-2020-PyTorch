#!/usr/bin/env bash

declare DATA_NAME="Credit"
declare TRAIN_DATA_PATH="./Datasets/${DATA_NAME}/train.csv"
declare SAMPLES_DIR="./Samples/${DATA_NAME}"
declare MODEL_SAVE_DIR="./Trained-Classifiers/${DATA_NAME}"

declare CLASSIFIER_NUM_HIDDEN_LAYERS="2"
declare CLASSIFIER_LEARNING_RATE="0.0001"
declare CLASSIFIER_BETA_1="0.9"
declare CLASSIFIER_BETA_2="0.999"
declare CLASSIFIER_BATCH_SIZE="1024"
declare CLASSIFIER_NUM_EPOCHS="100"
declare CLASSIFIER_RUN_DEVICE="cpu"

declare RAND_SEED="777"
declare VERBOSE="--verbose"

declare -a LIST_RATIO_BY_LABEL=("1=1.0" "1=2.0" "1=3.0" "1=4.0" "1=5.0")

python3 train.py \
    --train-data-path "${TRAIN_DATA_PATH}" \
    --model-save-path "${MODEL_SAVE_DIR}/imbalanced/classifier.pth" \
    --num-hidden-layers "${CLASSIFIER_NUM_HIDDEN_LAYERS}" \
    --learning-rate "${CLASSIFIER_LEARNING_RATE}" \
    --beta-1 "${CLASSIFIER_BETA_1}" \
    --beta-2 "${CLASSIFIER_BETA_2}" \
    --batch-size "${CLASSIFIER_BATCH_SIZE}" \
    --num-epochs "${CLASSIFIER_NUM_EPOCHS}" \
    --run-device "${CLASSIFIER_RUN_DEVICE}" \
    --rand-seed "${RAND_SEED}" \
    ${VERBOSE}

declare sampling_method="smote"
declare -a list_k_neighbors=(3 5 7)
for k_neighbors in ${list_k_neighbors[@]}; do
    for ratio_by_label in ${LIST_RATIO_BY_LABEL[@]}; do
        python3 train.py \
            --train-data-path "${TRAIN_DATA_PATH}" \
            --sampling-method "${sampling_method}" \
            --ratio-by-label "${ratio_by_label}" \
            --samples-dir "${SAMPLES_DIR}" \
            --smote-k-neighbors ${k_neighbors} \
            --model-save-path "${MODEL_SAVE_DIR}/${sampling_method}/k_neighbors=${k_neighbors}/ratio_by_label=${ratio_by_label}/classifier.pth" \
            --num-hidden-layers "${CLASSIFIER_NUM_HIDDEN_LAYERS}" \
            --learning-rate "${CLASSIFIER_LEARNING_RATE}" \
            --beta-1 "${CLASSIFIER_BETA_1}" \
            --beta-2 "${CLASSIFIER_BETA_2}" \
            --batch-size "${CLASSIFIER_BATCH_SIZE}" \
            --num-epochs "${CLASSIFIER_NUM_EPOCHS}" \
            --run-device "${CLASSIFIER_RUN_DEVICE}" \
            --rand-seed "${RAND_SEED}" \
            ${VERBOSE}
    done
done

declare sampling_method="smote_svm"
declare -a list_k_neighbors=(5 7)
declare -a list_svm_kernel=("linear" "poly" "rbf" "sigmoid")
for k_neighbors in ${list_k_neighbors[@]}; do
    for svm_kernel in ${list_svm_kernel[@]}; do
        for ratio_by_label in ${LIST_RATIO_BY_LABEL[@]}; do
            python3 train.py \
                --train-data-path "${TRAIN_DATA_PATH}" \
                --sampling-method "${sampling_method}" \
                --ratio-by-label "${ratio_by_label}" \
                --samples-dir "${SAMPLES_DIR}" \
                --smote-k-neighbors ${k_neighbors} \
                --smote-svm-kernel ${svm_kernel} \
                --model-save-path "${MODEL_SAVE_DIR}/${sampling_method}/k_neighbors=${k_neighbors}/svm_kernel=${svm_kernel}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --num-hidden-layers "${CLASSIFIER_NUM_HIDDEN_LAYERS}" \
                --learning-rate "${CLASSIFIER_LEARNING_RATE}" \
                --beta-1 "${CLASSIFIER_BETA_1}" \
                --beta-2 "${CLASSIFIER_BETA_2}" \
                --batch-size "${CLASSIFIER_BATCH_SIZE}" \
                --num-epochs "${CLASSIFIER_NUM_EPOCHS}" \
                --run-device "${CLASSIFIER_RUN_DEVICE}" \
                --rand-seed "${RAND_SEED}" \
                ${VERBOSE}
        done
    done
done

declare sampling_method="gan"
declare -a list_size_latent=(100 120)
declare -a list_num_hidden_layers=(2 4)
for size_latent in ${list_size_latent[@]}; do
    for num_hidden_layers in ${list_num_hidden_layers[@]}; do
        for ratio_by_label in ${LIST_RATIO_BY_LABEL[@]}; do
            python3 train.py \
                --train-data-path "${TRAIN_DATA_PATH}" \
                --sampling-method "${sampling_method}" \
                --ratio-by-label "${ratio_by_label}" \
                --samples-dir "${SAMPLES_DIR}" \
                --gan-size-latent ${size_latent} \
                --gan-num-hidden-layers ${num_hidden_layers} \
                --model-save-path "${MODEL_SAVE_DIR}/${sampling_method}/size_latent=${size_latent}/num_hidden_layers=${num_hidden_layers}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --num-hidden-layers "${CLASSIFIER_NUM_HIDDEN_LAYERS}" \
                --learning-rate "${CLASSIFIER_LEARNING_RATE}" \
                --beta-1 "${CLASSIFIER_BETA_1}" \
                --beta-2 "${CLASSIFIER_BETA_2}" \
                --batch-size "${CLASSIFIER_BATCH_SIZE}" \
                --num-epochs "${CLASSIFIER_NUM_EPOCHS}" \
                --run-device "${CLASSIFIER_RUN_DEVICE}" \
                --rand-seed "${RAND_SEED}" \
                ${VERBOSE}
        done
    done
done
