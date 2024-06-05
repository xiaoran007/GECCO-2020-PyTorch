#!/usr/bin/env bash

declare DATA_NAME="Credit"
declare DATA_PATH="./Datasets/${DATA_NAME}"

declare SAMPLES_DIR="./Samples/${DATA_NAME}"
declare MAX_RATIO="5.0"
declare RATIO_BY_LABEL="1=${MAX_RATIO}"

declare RAND_SEED="777"


declare sampling_method="gan"
declare -a list_size_latent=(100 120)
declare -a list_num_hidden_layers=(2 4)
for size_latent in ${list_size_latent[@]}; do
    for num_hidden_layers in ${list_num_hidden_layers[@]}; do
        python3 oversample.py \
            --data-path "${DATA_PATH}/train.csv" \
            --sampling-method "${sampling_method}" \
            --ratio-by-label "${RATIO_BY_LABEL}" \
            --gan-size-latent ${size_latent} \
            --gan-num-hidden-layers ${num_hidden_layers} \
            --save-path "${SAMPLES_DIR}/${sampling_method}/size_latent=${size_latent}/num_hidden_layers=${num_hidden_layers}/ratio_by_label=${RATIO_BY_LABEL}/sample_by_label.pkl" \
            --rand-seed 777
    done
done
