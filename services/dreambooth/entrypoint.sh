#!/bin/bash

set -Eeuo pipefail

[ -z "$INSTANCE_DIR" ] && echo "INSTANCE_DIR not specified" && exit 110 
#[ -z "$TEXTENCODER_STEPS" ] && echo "TEXTENCODER_STEPS not specified" && exit 120
[ -z "$MODEL_NAME" ] && echo "MODEL_NAME not specified" && exit 130
[ -z "$OUTPUT_DIR" ] && echo "OUTPUT_DIR not specified" && exit 140

mkdir -p "$OUTPUT_DIR"
SESSION_DIR="${OUTPUT_DIR}/${MODEL_NAME}"
SESSION_MODEL_DIR="${SESSION_DIR}/model"
UNET_FILE="${SESSION_MODEL_DIR}/unet/diffusion_pytorch_model.bin"
MODEL_DOWNLOADED="${SESSION_MODEL_DIR}/downloaded.ckpt"

if [ ! -f "$UNET_FILE" ]; then
  echo "Creating new session for ${MODEL_NAME}..."
  mkdir -p "$SESSION_DIR"
  rm -rf "$SESSION_MODEL_DIR"
  mkdir -p "$SESSION_MODEL_DIR"
  if [ -z "$MODEL_PATH" ]; then
    echo "Using the default model..."
    cp -r "/content/model" "${SESSION_MODEL_DIR}"
  elif [[ "$MODEL_PATH" = "/"* ]]; then
    echo "Using model at ${MODEL_PATH}"
    python3 -u /content/hf-diffusers/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "${MODEL_PATH}" --dump_path "${SESSION_MODEL_DIR}"
  elif [[ "$MODEL_PATH" = "http"* ]]; then
    echo "Downloading model from ${MODEL_PATH}"
    rm -f "$MODEL_DOWNLOADED"
    wget -O "$MODEL_DOWNLOADED" "$MODEL_PATH" || exit 210
    python3 -u /content/hf-diffusers/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "$MODEL_DOWNLOADED" --dump_path "$SESSION_MODEL_DIR"
    rm -f "$MODEL_DOWNLOADED"
  else
    echo "Invalid MODEL_PATH: ${MODEL_PATH}"
    exit 215
  fi
  if [ ! -f "$UNET_FILE" ]; then
    echo "Unable to find the model!"
    exit 220
  fi
else
  echo "Resuming previous session..."
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting Dreambooth training..."
accelerate launch /content/diffusers/examples/dreambooth/train_dreambooth.py \
  --image_captions_filename
  --train_text_encoder
  --save_starting_step=1000 \
  --stop_text_encoder_training=500 \
  --save_n_steps=500 \
  --pretrained_model_name_or_path="${SESSION_MODEL_DIR}" \
  --instance_data_dir="$INSTANCE_DIR" \
  --output_dir="$SESSION_DIR" \
  --instance_prompt="$MODEL_NAME" \
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

echo "Saving final CKPT..."
FINAL_CKPT="${SESSION_DIR}/${SESSION_NAME}.ckpt"
rm -f "${FINAL_CKPT}"
python3 /content/hf-diffusers/scripts/convert_diffusers_to_original_stable_diffusion.py --model_path "${SESSION_DIR}" --checkpoint_path "${FINAL_CKPT}" --half

#exec "$@"
