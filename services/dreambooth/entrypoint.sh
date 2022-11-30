#!/bin/bash

set -Eeuo pipefail


get_hf_sd_repo () {
  local REPO_NAME="$1"
  local DEST_DIR="$2"
  local HF_TOKEN="$3"
  local REPO_URL="https://huggingface.co/${REPO_NAME}"
  [[ ! -z "$HF_TOKEN" ]] && REPO_URL="https://USER:${HF_TOKEN}@huggingface.co/${REPO_NAME}"
  [ -f "${DEST_DIR}/unet/diffusion_pytorch_model.bin" ] && echo "Getting SD model from cache instead..." && return 0
  rm -rf "${DEST_DIR}"
  mkdir -p "${DEST_DIR}"
  cd "${DEST_DIR}"
  git init .
  git lfs install --system --skip-repo
  git remote add -f origin "$REPO_URL"
  git config core.sparsecheckout true
  echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json" > .git/info/sparse-checkout
  git pull origin main
  rm -rf ./.git
}

get_hf_vae_repo () {
  local REPO_NAME="$1"
  local DEST_DIR="$2"
  local REPO_URL="https://huggingface.co/${REPO_NAME}"
  [ -f "${DEST_DIR}/diffusion_pytorch_model.bin" ] && echo "Getting VAE model from cache instead..." && return 0
  rm -rf "${DEST_DIR}"
  mkdir -p "${DEST_DIR}"
  git clone "$REPO_URL" "${DEST_DIR}"
  rm -rf "${DEST_DIR}/.git"
}


## Disabled because it's non-deterministic, making models differ when trained on different machines.
#THREADS_COUNT=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')

echo "================== Environment Variables =================="
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
echo "SAVE_STARTING_STEPS=${SAVE_STARTING_STEPS}"
echo "SAVE_N_STEPS=${SAVE_N_STEPS}"
echo "SEED=${SEED}"
echo "INSTANCE_DIR=${INSTANCE_DIR}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "VAE_PATH=${VAE_PATH}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "KEEP_DIFFUSERS_MODEL=${KEEP_DIFFUSERS_MODEL}"
echo "SAVE_INTERMEDIARY_DIRS=${SAVE_INTERMEDIARY_DIRS}"
#echo "USE_BITSANDBYTES=${USE_BITSANDBYTES}"
echo "==========================================================="

[[ $MAX_TRAIN_STEPS -lt 100 ]] && MAX_TRAIN_STEPS=100 && echo "Setting MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
[[ $TEXT_ENCODER_STEPS -lt 0 ]] && TEXT_ENCODER_STEPS=0 && echo "Setting TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
[[ $TEXT_ENCODER_STEPS -gt $MAX_TRAIN_STEPS ]] && TEXT_ENCODER_STEPS=$MAX_TRAIN_STEPS && echo "Setting TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
[[ -z $SEED ]] && SEED=1337 && echo "Setting SEED=${SEED}"
[[ $SAVE_STARTING_STEPS -lt 0 ]] && SAVE_STARTING_STEPS=0 && echo "Setting SAVE_STARTING_STEPS=${SAVE_STARTING_STEPS}"
[[ $SAVE_N_STEPS -lt 100 ]] && SAVE_N_STEPS=100 && echo "Setting SAVE_N_STEPS=${SAVE_N_STEPS}"

mkdir -p "$CACHE_DIR"
mkdir -p "$OUTPUT_DIR"
SESSION_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
UNET_FILE="${SESSION_DIR}/unet/diffusion_pytorch_model.bin"
VAE_FILE="${SESSION_DIR}/vae/diffusion_pytorch_model.bin"
MODEL_DOWNLOADED="${SESSION_DIR}/downloaded.ckpt"

if [[ ! -f "$UNET_FILE" || ! -f "$VAE_FILE" ]]; then
  echo "Creating new session for ${MODEL_NAME}..."
  mkdir -p "$SESSION_DIR"
  find "${SESSION_DIR}/" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;
  rm -f "${SESSION_DIR}/model_index.json"
  rm -f "${SESSION_DIR}/v1-inference.yaml"

  # Copy SD model from repo or from CKPT file.
  if [[ "$MODEL_PATH" = "/"* && "$MODEL_PATH" = *".ckpt" ]]; then
    echo "Extracting from CKPT model at ${MODEL_PATH}"
    python3 -u /content/hf-diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "${MODEL_PATH}" --dump_path "$SESSION_DIR"
  elif [[ "$MODEL_PATH" = "http"* ]]; then
    echo "Downloading CKPT model from ${MODEL_PATH}"
    rm -f "$MODEL_DOWNLOADED"
    wget -O "$MODEL_DOWNLOADED" "$MODEL_PATH" || exit 210
    echo "Extracting from the downloaded CKPT model..."
    python3 -u /content/hf-diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "$MODEL_DOWNLOADED" --dump_path "$SESSION_DIR"
    rm -f "$MODEL_DOWNLOADED"
  else
    echo "Downloading base SD model from ${MODEL_PATH}..."
    get_hf_sd_repo "$MODEL_PATH" "${CACHE_DIR}/${MODEL_PATH}" "$HF_TOKEN"
    echo "Copying the base SD model to session directory..."
    rsync -ahq "${CACHE_DIR}/${MODEL_PATH}/" "${SESSION_DIR}/"
  fi

  # Replace VAE if specified to.
  if [ ! -z "$VAE_PATH" ]; then
    echo "Downloading base VAE model from ${VAE_PATH}..."
    get_hf_vae_repo "$VAE_PATH" "${CACHE_DIR}/${VAE_PATH}"
    echo "Replacing the base VAE model in the session directory..."
    rm -rf "${SESSION_DIR}/vae"
    if [ -f "${CACHE_DIR}/${VAE_PATH}/vae/diffusion_pytorch_model.bin" ]; then
      rsync -ahq "${CACHE_DIR}/${VAE_PATH}/vae/" "${SESSION_DIR}/vae/"
    else
      rsync -ahq "${CACHE_DIR}/${VAE_PATH}/" "${SESSION_DIR}/vae/"
    fi
  fi

  # Check if models were copied successfully
  if [[ -f "$UNET_FILE" && -f "$VAE_FILE" ]]; then
    echo "Found all model files."
  else
    find "${SESSION_DIR}/" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;
    rm -f "${SESSION_DIR}/model_index.json"
    rm -f "${SESSION_DIR}/v1-inference.yaml"
    echo "Unable to find the SD and/or VAE model!"
    exit 220
  fi
else
  echo "Resuming previous session ${MODEL_NAME}..."
fi

rm -f "${SESSION_DIR}/v1-inference.yaml"
mkdir -p "$OUTPUT_DIR"

echo "Starting Dreambooth training..."
echo "INSTANCE_DIR: $INSTANCE_DIR"
echo "SESSION_DIR: $SESSION_DIR"

dqt='"'
ARG_TEXT_ENCODER_STEPS="--train_text_encoder "
[[ $TEXT_ENCODER_STEPS -eq 0 ]] && ARG_TEXT_ENCODER_STEPS=""
ARG_USE_BITSANDBYTES="--use_8bit_adam "
#[[ $USE_BITSANDBYTES -eq 0 ]] && ARG_USE_BITSANDBYTES=""

RUN_TRAINING=$(cat << EOF
accelerate launch \
  --mixed_precision=fp16 \
  --num_processes=1 \
  --num_machines=1 \
  --num_cpu_threads_per_process=4 \
  /content/diffusers/examples/dreambooth/train_dreambooth.py \
    --image_captions_filename \
    ${ARG_TEXT_ENCODER_STEPS} \
    --save_intermediary_dirs=$SAVE_INTERMEDIARY_DIRS \
    --save_starting_step=$SAVE_STARTING_STEPS \
    --stop_text_encoder_training=$TEXT_ENCODER_STEPS \
    --save_n_steps=$SAVE_N_STEPS \
    --pretrained_model_name_or_path=$SESSION_DIR \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$SESSION_DIR \
    --Session_dir=$SESSION_DIR \
    --instance_prompt=$MODEL_NAME \
    --seed=$SEED \
    --resolution=512 \
    --mixed_precision=fp16 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    ${ARG_USE_BITSANDBYTES} \
    --learning_rate=2e-6 \
    --lr_scheduler=polynomial \
    --center_crop \
    --lr_warmup_steps=0 \
    --max_train_steps=$MAX_TRAIN_STEPS \
    --diffusers_to_ckpt_script_path='/content/hf-diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py'
EOF
)

eval $RUN_TRAINING

# Delete diffusers model if no flag to keep it.
if [[ $KEEP_DIFFUSERS_MODEL -eq 0 ]]; then
  find "${SESSION_DIR}/" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;
  rm -f "${SESSION_DIR}/model_index.json"
  rm -f "${SESSION_DIR}/v1-inference.yaml"
fi

#exec "$@"
