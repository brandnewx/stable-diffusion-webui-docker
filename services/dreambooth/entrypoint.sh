#!/bin/bash

set -Eeuo pipefail

THREADS_COUNT=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')

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
echo "KEEP_DIFFUSERS_MODEL=${KEEP_DIFFUSERS_MODEL}"
echo "SAVE_INTERMEDIARY_DIRS=${SAVE_INTERMEDIARY_DIRS}"
echo "THREADS_COUNT=${THREADS_COUNT}"
echo "==========================================================="

[[ $MAX_TRAIN_STEPS -lt 100 ]] && MAX_TRAIN_STEPS=100 && echo "Setting MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
[[ $TEXT_ENCODER_STEPS -lt 0 ]] && TEXT_ENCODER_STEPS=0 && echo "Setting TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
[[ $TEXT_ENCODER_STEPS -gt $MAX_TRAIN_STEPS ]] && TEXT_ENCODER_STEPS=$MAX_TRAIN_STEPS && echo "Setting TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
[[ -z $SEED ]] && SEED=1337 && echo "Setting SEED=${SEED}"
[[ $SAVE_STARTING_STEPS -lt 0 ]] && SAVE_STARTING_STEPS=0 && echo "Setting SAVE_STARTING_STEPS=${SAVE_STARTING_STEPS}"
[[ $SAVE_N_STEPS -lt 100 ]] && SAVE_N_STEPS=100 && echo "Setting SAVE_N_STEPS=${SAVE_N_STEPS}"

mkdir -p "$OUTPUT_DIR"
SESSION_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
UNET_FILE="${SESSION_DIR}/unet/diffusion_pytorch_model.bin"
MODEL_DOWNLOADED="${SESSION_DIR}/downloaded.ckpt"

if [ ! -f "$UNET_FILE" ]; then
  echo "Creating new session for ${MODEL_NAME}..."
  mkdir -p "$SESSION_DIR"
  find "${SESSION_DIR}/" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;
  rm -f "${SESSION_DIR}/model_index.json"
  rm -f "${SESSION_DIR}/v1-inference.yaml"
  if [ -z "$MODEL_PATH" ]; then
    echo "Using the default model..."
    rsync -ahq "/content/model/" "${SESSION_DIR}/"
  elif [[ "$MODEL_PATH" = "/"* ]]; then
    echo "Using model at ${MODEL_PATH}"
    python3 -u /content/hf-diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "${MODEL_PATH}" --dump_path "$SESSION_DIR"
  elif [[ "$MODEL_PATH" = "http"* ]]; then
    echo "Downloading model from ${MODEL_PATH}"
    rm -f "$MODEL_DOWNLOADED"
    wget -O "$MODEL_DOWNLOADED" "$MODEL_PATH" || exit 210
    python3 -u /content/hf-diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "$MODEL_DOWNLOADED" --dump_path "$SESSION_DIR"
    rm -f "$MODEL_DOWNLOADED"
  else
    echo "Invalid MODEL_PATH: ${MODEL_PATH}"
    exit 215
  fi
  if [ ! -f "$UNET_FILE" ]; then
    find "${SESSION_DIR}/" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;
    rm -f "${SESSION_DIR}/model_index.json"
    rm -f "${SESSION_DIR}/v1-inference.yaml"
    echo "Unable to find the model!"
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

if [[ $TEXT_ENCODER_STEPS -gt 0 ]]; then
  echo "Starts training with TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
  accelerate launch \
    --mixed_precision=fp16 \
    --num_processes=1 \
    --num_machines=1 \
    --num_cpu_threads_per_process=$THREADS_COUNT \
    /content/diffusers/examples/dreambooth/train_dreambooth.py \
      --image_captions_filename \
      --train_text_encoder \
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
      --mixed_precision="fp16" \
      --train_batch_size=1 \
      --gradient_accumulation_steps=1 \
      --use_8bit_adam \
      --learning_rate=2e-6 \
      --lr_scheduler="polynomial" \
      --center_crop \
      --lr_warmup_steps=0 \
      --max_train_steps=$MAX_TRAIN_STEPS \
      --diffusers_to_ckpt_script_path="/content/hf-diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py"
else
  echo "Starts training without training the text encoder. TEXT_ENCODER_STEPS=${TEXT_ENCODER_STEPS}"
  accelerate launch \
    --mixed_precision=fp16 \
    --num_processes=1 \
    --num_machines=1 \
    --num_cpu_threads_per_process=$THREADS_COUNT \
    /content/diffusers/examples/dreambooth/train_dreambooth.py \
      --image_captions_filename \
      --save_intermediary_dirs=$SAVE_INTERMEDIARY_DIRS \
      --save_starting_step=$SAVE_STARTING_STEPS \
      --stop_text_encoder_training=0 \
      --save_n_steps=$SAVE_N_STEPS \
      --pretrained_model_name_or_path=$SESSION_DIR \
      --instance_data_dir=$INSTANCE_DIR \
      --output_dir=$SESSION_DIR \
      --Session_dir=$SESSION_DIR \
      --instance_prompt=$MODEL_NAME \
      --seed=$SEED \
      --resolution=512 \
      --mixed_precision="fp16" \
      --train_batch_size=1 \
      --gradient_accumulation_steps=1 \
      --use_8bit_adam \
      --learning_rate=2e-6 \
      --lr_scheduler="polynomial" \
      --center_crop \
      --lr_warmup_steps=0 \
      --max_train_steps=$MAX_TRAIN_STEPS \
      --diffusers_to_ckpt_script_path="/content/hf-diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py"
fi


# Delete diffusers model if no flag to keep it.
if [[ $KEEP_DIFFUSERS_MODEL -eq 0 ]]; then
  find "${SESSION_DIR}/" -maxdepth 1 -mindepth 1 -type d -exec rm -rf {} \;
  rm -f "${SESSION_DIR}/model_index.json"
  rm -f "${SESSION_DIR}/v1-inference.yaml"
fi

#exec "$@"
