#!/bin/bash

set -Eeuo pipefail


get_hf_sd_repo () {
  local REPO_NAME="$1"
  local DEST_DIR="$2"
  local HF_TOKEN="$3"
  local COMMIT="$4"
  local REPO_URL="https://huggingface.co/${REPO_NAME}"
  [[ ! -z "$HF_TOKEN" ]] && REPO_URL="https://USER:${HF_TOKEN}@huggingface.co/${REPO_NAME}"
  [ -f "${DEST_DIR}/unet/diffusion_pytorch_model.bin" ] && echo "Getting SD model from cache instead..." && return 0
  echo "Make sure you accepted the terms in ${REPO_URL}"
  rm -rf "${DEST_DIR}"
  mkdir -p "${DEST_DIR}"
  cd "${DEST_DIR}"
  git init .
  git lfs install --system --skip-repo
  git remote add -f origin "$REPO_URL"
  git config core.sparsecheckout true
  echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json" > .git/info/sparse-checkout
  git reset --hard $COMMIT
  git fetch
  rm -rf ./.git
}

get_hf_vae_repo () {
  local REPO_NAME="$1"
  local DEST_DIR="$2"
  local COMMIT="$3"
  local REPO_URL="https://huggingface.co/${REPO_NAME}"
  [ -f "${DEST_DIR}/diffusion_pytorch_model.bin" ] && echo "Getting VAE model from cache instead..." && return 0
  rm -rf "${DEST_DIR}"
  mkdir -p "${DEST_DIR}"
  cd "${DEST_DIR}"
  git clone "$REPO_URL" .
  git reset --hard $COMMIT
  rm -rf "${DEST_DIR}/.git"
}


## Disabled because it's non-deterministic, making models differ when trained on different machines.
#THREADS_COUNT=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')

echo "================== Environment Variables =================="
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
echo "SAVE_STARTING_STEPS=${SAVE_STARTING_STEPS}"
echo "SAVE_N_STEPS=${SAVE_N_STEPS}"
echo "SEED=${SEED}"
echo "INSTANCE_DIR=${INSTANCE_DIR}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "MODEL_COMMIT=${MODEL_COMMIT}"
echo "VAE_PATH=${VAE_PATH}"
echo "VAE_COMMIT=${VAE_COMMIT}"
echo "CACHE_DIR=${CACHE_DIR}"
echo "KEEP_DIFFUSERS_MODEL=${KEEP_DIFFUSERS_MODEL}"
echo "SAVE_INTERMEDIARY_DIRS=${SAVE_INTERMEDIARY_DIRS}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "SCHEDULER_NAME=${SCHEDULER_NAME}"
echo "SUBFOLDER_MODE=${SUBFOLDER_MODE}"
echo "CLASSES_ROOT_DIR=${CLASSES_ROOT_DIR}"
echo "NUM_CLASS_IMAGES=${NUM_CLASS_IMAGES}"
#echo "USE_BITSANDBYTES=${USE_BITSANDBYTES}"
echo "==========================================================="

[[ $MAX_TRAIN_STEPS -lt 100 ]] && MAX_TRAIN_STEPS=100 && echo "Setting MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
[[ $TEXT_ENCODER_STEPS -lt 0 ]] && TEXT_ENCODER_STEPS=0 && echo "Setting TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
[[ $TEXT_ENCODER_STEPS -gt $MAX_TRAIN_STEPS ]] && TEXT_ENCODER_STEPS=$MAX_TRAIN_STEPS && echo "Setting TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
[[ -z $SEED ]] && SEED=1337 && echo "Setting SEED=${SEED}"
[[ $SAVE_STARTING_STEPS -lt 0 ]] && SAVE_STARTING_STEPS=0 && echo "Setting SAVE_STARTING_STEPS=${SAVE_STARTING_STEPS}"
[[ $SAVE_N_STEPS -lt 100 ]] && SAVE_N_STEPS=100 && echo "Setting SAVE_N_STEPS=${SAVE_N_STEPS}"
[[ -z $MODEL_COMMIT ]] && MODEL_COMMIT="main" && echo "Setting MODEL_COMMIT=${MODEL_COMMIT}"
[[ -z $VAE_COMMIT ]] && VAE_COMMIT="main" && echo "Setting VAE_COMMIT=${VAE_COMMIT}"

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
    python3 -u /content/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "${MODEL_PATH}" --dump_path "$SESSION_DIR" --clipvit_path "/content/models/clip-vit-large-patch14" --bert_path "/content/models/bert-base-uncased" --tokenizer_path "/content/models/tokenizer"
  elif [[ "$MODEL_PATH" = "http"* ]]; then
    echo "Downloading CKPT model from ${MODEL_PATH}"
    rm -f "$MODEL_DOWNLOADED"
    wget -O "$MODEL_DOWNLOADED" "$MODEL_PATH" || exit 210
    echo "Extracting from the downloaded CKPT model..."
    python3 -u /content/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "$MODEL_DOWNLOADED" --dump_path "$SESSION_DIR" --clipvit_path "/content/models/clip-vit-large-patch14" --bert_path "/content/models/bert-base-uncased" --tokenizer_path "/content/models/tokenizer"
    rm -f "$MODEL_DOWNLOADED"
  else
    echo "Downloading base SD model from ${MODEL_PATH}..."
    get_hf_sd_repo "$MODEL_PATH" "${CACHE_DIR}/${MODEL_PATH}/${MODEL_COMMIT}" "$HF_TOKEN" "$MODEL_COMMIT"
    echo "Copying the base SD model to session directory..."
    rm -rf "${CACHE_DIR}/${MODEL_PATH}/${MODEL_COMMIT}/.git"
    rsync -ahq "${CACHE_DIR}/${MODEL_PATH}/${MODEL_COMMIT}/" "${SESSION_DIR}/"
  fi

  # Replace VAE if specified to.
  if [ ! -z "$VAE_PATH" ]; then
    echo "Downloading base VAE model from ${VAE_PATH}..."
    get_hf_vae_repo "$VAE_PATH" "${CACHE_DIR}/${VAE_PATH}/${VAE_COMMIT}" "$VAE_COMMIT"
    echo "Replacing the base VAE model in the session directory..."
    rm -rf "${SESSION_DIR}/vae"
    if [ -f "${CACHE_DIR}/${VAE_PATH}/${VAE_COMMIT}/vae/diffusion_pytorch_model.bin" ]; then
      rm -rf "${CACHE_DIR}/${VAE_PATH}/${VAE_COMMIT}/vae/.git"
      rsync -ahq "${CACHE_DIR}/${VAE_PATH}/${VAE_COMMIT}/vae/" "${SESSION_DIR}/vae/"
    else
      rm -rf "${CACHE_DIR}/${VAE_PATH}/${VAE_COMMIT}/.git"
      rsync -ahq "${CACHE_DIR}/${VAE_PATH}/${VAE_COMMIT}/" "${SESSION_DIR}/vae/"
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

if [[ -z $SUBFOLDER_MODE || $SUBFOLDER_MODE -le 0 ]]; then
  accelerate launch \
    --mixed_precision=fp16 \
    --num_processes=4 \
    --num_machines=1 \
    --num_cpu_threads_per_process=1 \
    /content/diffusers/examples/dreambooth/train_dreambooth.py \
      --image_captions_filename \
      --train_text_encoder \
      --use_8bit_adam \
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
      --learning_rate=$LEARNING_RATE \
      --scale_lr \
      --lr_scheduler=constant \
      --center_crop \
      --lr_warmup_steps=0 \
      --max_train_steps=$MAX_TRAIN_STEPS \
      --scheduler_name=$SCHEDULER_NAME \
      --diffusers_to_ckpt_script_path="/content/diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py"
else
  accelerate launch \
    --mixed_precision=fp16 \
    --num_processes=4 \
    --num_machines=1 \
    --num_cpu_threads_per_process=1 \
    /content/diffusers/examples/dreambooth/train_dreambooth.py \
      --gradient_checkpointing \
      --image_captions_filename \
      --train_text_encoder \
      --with_prior_preservation \
      --prior_loss_weight=1.0 \
      --subfolder_mode \
      --class_data_dir=$CLASSES_ROOT_DIR \
      --num_class_images=$NUM_CLASS_IMAGES \
      --use_8bit_adam \
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
      --learning_rate=$LEARNING_RATE \
      --scale_lr \
      --lr_scheduler=constant \
      --center_crop \
      --lr_warmup_steps=0 \
      --max_train_steps=$MAX_TRAIN_STEPS \
      --scheduler_name=$SCHEDULER_NAME \
      --diffusers_to_ckpt_script_path="/content/diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py"
fi

# Delete diffusers model if no flag to keep it.
if [[ $KEEP_DIFFUSERS_MODEL -eq 0 ]]; then
  find "${SESSION_DIR}/" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;
  rm -f "${SESSION_DIR}/model_index.json"
  rm -f "${SESSION_DIR}/v1-inference.yaml"
fi

# Save debug log
[ -d "/usr/local/lib/python3.10/dist-packages" ] && echo $(ls -a /usr/local/lib/python3.10/dist-packages/ | sort) > "${SESSION_DIR}/dist-packages.log"
[ -d "/usr/local/lib/python3.11/dist-packages" ] && echo $(ls -a /usr/local/lib/python3.11/dist-packages/ | sort) > "${SESSION_DIR}/dist-packages1.log"
[ -d "/usr/local/lib/python3.12/dist-packages" ] && echo $(ls -a /usr/local/lib/python3.12/dist-packages/ | sort) > "${SESSION_DIR}/dist-packages2.log"
[ -f "/content/diffusers/examples/dreambooth/train_dreambooth.py" ] && cp -f "/content/diffusers/examples/dreambooth/train_dreambooth.py" "${SESSION_DIR}/train_dreambooth.py.log"
[ -f "/content/diffusers/setup.py" ] && cp -f "/content/diffusers/setup.py" "${SESSION_DIR}/setup.py.log"
[ -f "/content/diffusers/examples/dreambooth/requirements.txt" ] && cp -f "/content/diffusers/examples/dreambooth/requirements.txt" "${SESSION_DIR}/requirements.txt.log"
[ -f "/content/diffusers/tests/test_config.py" ] && cp -f "/content/diffusers/tests/test_config.py" "${SESSION_DIR}/test_config.py.log"

# Move checkpoints to the checkpoint directory
if [ ! -z $CHECKPOINT_DIR ]; then
  echo "Moving checkponts to the checkpoint directory..."
  mkdir -p "${CHECKPOINT_DIR}/${MODEL_NAME}"
  rsync -ah --remove-source-files \
    --include="*.ckpt" \
    --include="*.safetensors" \
    --include="*.log" \
    --exclude="*" \
    "${SESSION_DIR}/" "${CHECKPOINT_DIR}/${MODEL_NAME}/"
fi

# Clean up output directory
[ -z "$(ls -A $SESSION_DIR)" ] && rm -r "${SESSION_DIR}"

#exec "$@"
