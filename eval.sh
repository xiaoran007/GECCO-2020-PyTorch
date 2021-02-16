#!/usr/bin/env bash

declare DATA_NAME="Credit"
declare EVAL_DATA_PATH="./Datasets/${DATA_NAME}/eval.csv"
declare MODEL_LOAD_DIR="./Trained-Classifiers/${DATA_NAME}"

declare METRIC="f1_score"
declare CLASSIFIER_RUN_DEVICE="cpu"

declare -a LIST_RATIO_BY_LABEL=("1=1.0" "1=2.0" "1=3.0" "1=4.0" "1=5.0")

python3 eval.py \
    --eval-data-path "${EVAL_DATA_PATH}" \
    --model-load-path "${MODEL_LOAD_DIR}/imbalanced/classifier.pth" \
    --metric "${METRIC}" \
    --run-device "${CLASSIFIER_RUN_DEVICE}"

declare sampling_method="smote"
declare -a list_k_neighbors=(3 5 7)
for k_neighbors in ${list_k_neighbors[@]}; do
    for ratio_by_label in ${LIST_RATIO_BY_LABEL[@]}; do
        python3 eval.py \
            --eval-data-path "${EVAL_DATA_PATH}" \
            --model-load-path "${MODEL_LOAD_DIR}/${sampling_method}/k_neighbors=${k_neighbors}/ratio_by_label=${ratio_by_label}/classifier.pth" \
            --metric "${METRIC}" \
            --run-device "${CLASSIFIER_RUN_DEVICE}"
    done
done

declare sampling_method="smote_svm"
declare -a list_k_neighbors=(5 7)
declare -a list_svm_kernel=("linear" "poly" "rbf" "sigmoid")
for k_neighbors in ${list_k_neighbors[@]}; do
    for svm_kernel in ${list_svm_kernel[@]}; do
        for ratio_by_label in ${LIST_RATIO_BY_LABEL[@]}; do
            python3 eval.py \
                --eval-data-path "${EVAL_DATA_PATH}" \
                --model-load-path "${MODEL_LOAD_DIR}/${sampling_method}/k_neighbors=${k_neighbors}/svm_kernel=${svm_kernel}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --metric "${METRIC}" \
                --run-device "${CLASSIFIER_RUN_DEVICE}"
        done
    done
done

declare sampling_method="gan"
declare -a list_size_latent=(100 120)
declare -a list_num_hidden_layers=(2 4)
for size_latent in ${list_size_latent[@]}; do
    for num_hidden_layers in ${list_num_hidden_layers[@]}; do
        for ratio_by_label in ${LIST_RATIO_BY_LABEL[@]}; do
            python3 eval.py \
                --eval-data-path "${EVAL_DATA_PATH}" \
                --model-load-path "${MODEL_LOAD_DIR}/${sampling_method}/size_latent=${size_latent}/num_hidden_layers=${num_hidden_layers}/ratio_by_label=${ratio_by_label}/classifier.pth" \
                --metric "${METRIC}" \
                --run-device "${CLASSIFIER_RUN_DEVICE}"
        done
    done
done
