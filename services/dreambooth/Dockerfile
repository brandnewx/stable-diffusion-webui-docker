# syntax=docker/dockerfile:1

FROM brandnewx/ubuntu22:cuda-pytorch
#FROM brandnewx/ubuntu22:full
USER root
SHELL ["/bin/bash", "-ceuxo", "pipefail"]
WORKDIR /

RUN apt-get update -q && apt-get install -y git git-lfs && apt-get clean

## Copy default model
# COPY --from=download /content/model /content/model

# Install xformers
COPY --from=brandnewx/xformers /xformers-dist /xformers-dist
RUN pip uninstall xformers --yes && pip install /xformers-dist/*.whl && rm -rf /xformers-dist

RUN <<EOF
mkdir -p "/content/models/clip-vit-large-patch14"
cd "/content/models/clip-vit-large-patch14"
git init .
git lfs install --system --skip-repo
git remote add -f origin "https://huggingface.co/openai/clip-vit-large-patch14"
git config core.sparsecheckout true
echo -e "config.json\nmerges.txt\npreprocessor_config.json\npytorch_model.bin\nspecial_tokens_map.json\ntokenizer.json\ntokenizer_config.json\nvocab.json\nvocab.txt" > .git/info/sparse-checkout
git pull origin main
rm -rf ./.git
EOF

RUN <<EOF
mkdir -p "/content/models/bert-base-uncased"
cd "/content/models/bert-base-uncased"
git init .
git lfs install --system --skip-repo
git remote add -f origin "https://huggingface.co/bert-base-uncased"
git config core.sparsecheckout true
echo -e "config.json\nmerges.txt\npreprocessor_config.json\npytorch_model.bin\nspecial_tokens_map.json\ntokenizer.json\ntokenizer_config.json\nvocab.json\nvocab.txt" > .git/info/sparse-checkout
git pull origin main
rm -rf ./.git
EOF

RUN <<EOF
pip uninstall bitsandbytes --yes
pip install bitsandbytes==0.35.0 omegaconf==2.2.3
EOF

# triton==2.0.0.dev20221117 \
# accelerate==0.14.0 \
# open-clip-torch==2.7.0 \
# torchsde==0.2.5 \
# pytorch-lightning==1.8.3.post1

#ENV THELASTBEN_DIFFUSERS_COMMIT=70179b7bf3d6aa69994e1b0ffdeae440c7f054e5
ENV THELASTBEN_DIFFUSERS_COMMIT=main
ENV HF_DIFFUSERS_COMMIT=9f10c545cbf54dd4d87e7e0f24e1ec02e928c966

RUN <<EOF
echo "Installing Diffusers libraries..."
mkdir -p /content/diffusers
cd /content/diffusers
git clone https://github.com/brandnewx/thelastben .
echo "resetting to main #22222247..."
#git checkout pre-merge
git reset --hard ${THELASTBEN_DIFFUSERS_COMMIT}
python3 -u setup.py install
rm -rf ./.git
pip install -r /content/diffusers/examples/dreambooth/requirements.txt
EOF

WORKDIR /

COPY ./tokenizer /content/models/tokenizer

COPY ./entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]
CMD python3 --version
