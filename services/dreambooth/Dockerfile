FROM brandnewx/ubuntu22:cuda-pytorch
USER root
WORKDIR /

ARG HF_TOKEN=""
ARG CKPT_PATH=""

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

RUN <<EOF
  echo "Installing Diffusers libraries..."
  mkdir -p /content/hf-diffusers
  cd /content/hf-diffusers
  git clone https://github.com/huggingface/diffusers .
  rm -rf ./.git
  python3 -u setup.py

  mkdir -p /content/diffusers
  cd /content/diffusers
  git clone https://github.com/TheLastBen/diffusers .
  git reset --hard ac2141e9ad5b470c95567472adad186fbf254228
  python3 -u setup.py
EOF

RUN pip install triton==2.0.0.dev20221117 \
  accelerate==0.12.0 \
  omegaconf==2.2.3

# Install xformers
COPY --from=brandnewx/xformers /xformers-dist /xformers-dist
RUN pip uninstall xformers --yes && pip install /xformers-dist/*.whl && rm -rf /xformers-dist

# Download SD 1.5
RUN <<EOF
if [ $HF_TOKEN != "" ]; then
    echo "Downloading SD 1.5 model (Unet and Text Encoder)..."
    mkdir /content/model
    cd /content/model
    git init
    git lfs install --system --skip-repo
    git remote add -f origin "https://USER:${HF_TOKEN}@huggingface.co/runwayml/stable-diffusion-v1-5"
    git config core.sparsecheckout true
    echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nmodel_index.json" > .git/info/sparse-checkout
    git pull origin main
    if [ ! -f '/content/model/unet/diffusion_pytorch_model.bin' ]; then
        echo 'Make sure you accepted the terms in https://huggingface.co/runwayml/stable-diffusion-v1-5'
        return 1
    fi

    echo "Downloading SD 1.5 VAE..."
    mkdir -p /content/model/vae
    cd /content/model/vae
    git clone "https://USER:${HF_TOKEN}@huggingface.co/stabilityai/sd-vae-ft-mse" .
    rm -r ./.git
else
    echo "HF_TOKEN not specified. Skipped downloading SD model."
fi

if [ ${CKPT_PATH} != "" ]; then
    python3 -u /content/hf-diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "${CKPT_Path}" --dump_path /content/model
    if [ ! -f '/content/model/unet/diffusion_pytorch_model.bin' ]; then
        echo "Conversion error, Insufficient RAM or corrupt CKPT, use a 4.7GB CKPT instead of 7GB"
        return 1
    fi
fi
EOF

# Test initial run to install additional files.
RUN pip uninstall bitsandbytes --yes && pip install bitsandbytes==0.35.0

# WORKDIR ${ROOT}/repositories/stable-diffusion
WORKDIR ${ROOT}
ENV CLI_ARGS=""
EXPOSE 7860
ENTRYPOINT ["/docker/entrypoint.sh"]
# run, -u to not buffer stdout / stderr
CMD python3 -u ${ROOT}/webui.py --listen --port 7860 --ckpt-dir ${ROOT}/models/Stable-diffusion ${CLI_ARGS}
