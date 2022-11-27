# syntax=docker/dockerfile:1

FROM brandnewx/ubuntu22:cuda-pytorch as download
#FROM brandnewx/ubuntu22:full as download
USER root
SHELL ["/bin/bash", "-ceuxo", "pipefail"]
WORKDIR /

ARG HF_TOKEN
ARG HF_SD_REPO="runwayml/stable-diffusion-v1-5"
ARG HF_VAE_REPO="stabilityai/sd-vae-ft-mse" 

RUN apt-get update -q && apt-get install -y git git-lfs && apt-get clean

# Download model
RUN <<EOF
if [ -z "${HF_TOKEN}" ]; then
    echo "HF_TOKEN not specified."
    return 1
fi

echo "Downloading SD model (Unet and Text Encoder)..."
mkdir -p /content/model
cd /content/model
git init .
git lfs install --system --skip-repo
git remote add -f origin "https://USER:${HF_TOKEN}@huggingface.co/${HF_SD_REPO}"
git config core.sparsecheckout true
echo -e "feature_extractor\nsafety_checker\nscheduler\ntext_encoder\ntokenizer\nunet\nvae\nmodel_index.json" > .git/info/sparse-checkout
git pull origin main
rm -rf ./.git
EOF

RUN <<EOF
if [ ! -f '/content/model/unet/diffusion_pytorch_model.bin' ]; then
    echo "Make sure you accepted the terms in https://huggingface.co/${HF_SD_REPO}"
    return 2
fi
EOF

RUN <<EOF
# Optionally download and replace VAE
if [ ! -z "${HF_VAE_REPO}" ]; then
    echo "Downloading alternative VAE..."
    rm -rf /content/model/vae
    mkdir -p /content/model/vae
    cd /content/model/vae
    git clone "https://USER:${HF_TOKEN}@huggingface.co/${HF_VAE_REPO}" .
    rm -rf ./.git
fi
EOF

RUN <<EOF
if [ ! -f "/content/model/vae/diffusion_pytorch_model.bin" ]; then
    echo "VAE model is missing!"
    return 3
fi
EOF


FROM brandnewx/ubuntu22:cuda-pytorch
#FROM brandnewx/ubuntu22:full
USER root
SHELL ["/bin/bash", "-ceuxo", "pipefail"]
WORKDIR /

RUN apt-get update -q && apt-get install -y git git-lfs && apt-get clean

# Copy default model
COPY --from=download /content/model /content/model

# Install xformers
COPY --from=brandnewx/xformers /xformers-dist /xformers-dist
RUN pip uninstall xformers --yes && pip install /xformers-dist/*.whl && rm -rf /xformers-dist

RUN <<EOF
pip install triton==2.0.0.dev20221117 \
    accelerate==0.12.0 \
    omegaconf==2.2.3 \
    open-clip-torch==2.7.0 \
    torchsde==0.2.5 \
    pytorch-lightning==1.8.3.post1
pip uninstall bitsandbytes --yes
pip install bitsandbytes==0.35.0
EOF

ENV THELASTBEN_DIFFUSERS_COMMIT=69a5228ab2657a246154b8f3c1806044d121c07b
#ENV THELASTBEN_DIFFUSERS_COMMIT=main
ENV HF_DIFFUSERS_COMMIT=9f10c545cbf54dd4d87e7e0f24e1ec02e928c966

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
    git clone https://github.com/brandnewx/thelastben-diffusers .
    git reset --hard ${THELASTBEN_DIFFUSERS_COMMIT}
    python3 -u setup.py install
    rm -rf ./.git
EOF

# Install Dreambooth
RUN pip install -r /content/diffusers/examples/dreambooth/requirements.txt

WORKDIR /

COPY ./entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]
CMD python3 --version
