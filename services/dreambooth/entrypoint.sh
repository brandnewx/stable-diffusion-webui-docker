#!/bin/bash

set -Eeuo pipefail

[[ -z "$INSTANCE_DIR" ]] && echo "INSTANCE_DIR not specified" && return 10 
[[ -z "$TEXTENCODER_STEPS" ]] && echo "TEXTENCODER_STEPS not specified" && return 11
[[ -z "$SESSION_NAME" ]] && echo "SESSION_NAME not specified" && return 12

SESSION_DIR="/sessions/${SESSION_NAME}"
WORKING_MODEL_DIR="${SESSION_DIR}/working/model"
OUTPUT_DIR="${SESSION_DIR}/output"
MODEL_FILE="${WORKING_MODEL_DIR}/unet/diffusion_pytorch_model.bin"
MODEL_DOWNLOADED="${WORKING_MODEL_DIR}/downloaded.ckpt"

if [[ ! -f "${MODEL_FILE}" ]]; then
  echo "Creating new session..."
  mkdir -p "${SESSION_DIR}"
  rm -rf "${WORKING_MODEL_DIR}"
  mkdir -p "${WORKING_MODEL_DIR}"
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "Using the default model..."
    cp -r "/content/model" "${WORKING_MODEL_DIR}"
  elif [[ "${MODEL_PATH}" = "/"* ]]; then
    echo "Using model at ${MODEL_PATH}"
    python3 -u /content/hf-diffusers/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "${MODEL_PATH}" --dump_path "${WORKING_MODEL_DIR}"
  elif [[ "${MODEL_PATH}" = "https://"* ]]; then
    echo "Downloading model from ${MODEL_PATH}"
    rm -f "${MODEL_DOWNLOADED}"
    wget -O "${MODEL_DOWNLOADED}" ${MODEL_PATH} || return 13
    python3 -u /content/hf-diffusers/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "${MODEL_DOWNLOADED}" --dump_path "${WORKING_MODEL_DIR}"
  fi
  if [[ ! -f "${MODEL_FILE}" ]]; then
    echo "Unable to find the base model!"
    return 14
  fi
else
  echo "Resuming previous session..."
fi

mkdir -p "${OUTPUT_DIR}"

!accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
  --image_captions_filename
  --train_text_encoder
  --save_starting_step=500 \
  --stop_text_encoder_training=500 \
  --save_n_steps=500 \
  --pretrained_model_name_or_path="${WORKING_MODEL_DIR}" \
  --instance_data_dir="${INSTANCE_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --instance_prompt="${SESSION_NAME}" \
  --seed=1337 \
  --resolution=512 \
  --mixed_precision=fp16 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --learning_rate=2e-6 \
  --lr_scheduler="polynomial" \
  --center_crop \
  --lr_warmup_steps=0 \
  --max_train_steps=2000

#exec "$@"