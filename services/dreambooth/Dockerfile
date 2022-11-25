# syntax=docker/dockerfile:1

ARG HF_TOKEN="" \
    HF_SD_REPO="runwayml/stable-diffusion-v1-5" \
    HF_VAE_REPO="stabilityai/sd-vae-ft-mse" \
    CKPT_PATH=""

#FROM brandnewx/ubuntu22:cuda-pytorch
FROM brandnewx/ubuntu22:base
USER root
WORKDIR /

ENV THELASTBEN_DIFFUSERS_COMMIT=ac2141e9ad5b470c95567472adad186fbf254228 \
    HF_DIFFUSERS_COMMIT=9f10c545cbf54dd4d87e7e0f24e1ec02e928c966

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

RUN <<EOF
#  echo "Installing Diffusers libraries..."
#  mkdir -p /content/hf-diffusers
#  cd /content/hf-diffusers
#  git clone https://github.com/huggingface/diffusers .
#  git reset --hard ${HF_DIFFUSERS_COMMIT}
#  python3 -u setup.py install
#  rm -rf ./.git

    mkdir -p /content/hf-diffusers/scripts
    cd /content/hf-diffusers/scripts
    wget https://raw.githubusercontent.com/huggingface/diffusers/${HF_DIFFUSERS_COMMIT}/scripts/convert_original_stable_diffusion_to_diffusers.py
    wget https://raw.githubusercontent.com/huggingface/diffusers/${HF_DIFFUSERS_COMMIT}/scripts/convert_diffusers_to_original_stable_diffusion.py

    mkdir -p /content/diffusers
    cd /content/diffusers
    git clone https://github.com/TheLastBen/diffusers .
    git reset --hard ${THELASTBEN_DIFFUSERS_COMMIT}
    python3 -u setup.py install
    rm -rf ./.git
EOF

RUN pip install triton==2.0.0.dev20221117 \
  accelerate==0.12.0 \
  omegaconf==2.2.3

# Install xformers
#COPY --from=brandnewx/xformers /xformers-dist /xformers-dist
#RUN pip uninstall xformers --yes && pip install /xformers-dist/*.whl && rm -rf /xformers-dist

# Download SD 1.5
RUN <<EOF
if [ ! -z "${HF_TOKEN}" ]; then
    echo "Downloading SD 1.5 model (Unet and Text Encoder)..."
    mkdir /content/model
    cd /content/model
    git init
    git lfs install --system --skip-repo
    git remote add -f origin "https://USER:${HF_TOKEN}@huggingface.co/${HF_SD_REPO}"
    git config core.sparsecheckout true
    echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nmodel_index.json" > .git/info/sparse-checkout
    git pull origin main

    if [ ! -f '/content/model/unet/diffusion_pytorch_model.bin' ]; then
        echo "Make sure you accepted the terms in https://huggingface.co/${HF_SD_REPO}"
        return 1
    fi

    if [ ! -z "${HF_VAE_REPO}" ]; then
        echo "Downloading SD 1.5 VAE..."
        mkdir -p /content/model/vae
        cd /content/model/vae
        git clone "https://USER:${HF_TOKEN}@huggingface.co/${HF_VAE_REPO}" .
        rm -r ./.git
    fi
else
    echo "HF_TOKEN not specified. Skipped downloading SD model."

    if [ ! -z "${CKPT_PATH}" ]; then
        python3 -u /content/hf-diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "${CKPT_Path}" --dump_path /content/model
        if [ ! -f '/content/model/unet/diffusion_pytorch_model.bin' ]; then
            echo "Conversion error, Insufficient RAM or corrupt CKPT, use a 4.7GB CKPT instead of 7GB"
            return 1
        fi
    fi
fi

EOF

# Test initial run to install additional files.
RUN pip uninstall bitsandbytes --yes && pip install bitsandbytes==0.35.0

WORKDIR /
EXPOSE 7860
ENTRYPOINT ["entrypoint.sh"]
CMD python3 --version
