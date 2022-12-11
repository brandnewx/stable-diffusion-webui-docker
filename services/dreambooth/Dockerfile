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
mkdir -p /content/hf-diffusers/scripts
cd /content/hf-diffusers/scripts
wget https://raw.githubusercontent.com/huggingface/diffusers/${HF_DIFFUSERS_COMMIT}/scripts/convert_original_stable_diffusion_to_diffusers.py
wget https://raw.githubusercontent.com/huggingface/diffusers/${HF_DIFFUSERS_COMMIT}/scripts/convert_diffusers_to_original_stable_diffusion.py

mkdir -p /content/diffusers
cd /content/diffusers
git clone https://github.com/brandnewx/thelastben .
echo "resetting to main #22222245..."
#git checkout pre-merge
git reset --hard ${THELASTBEN_DIFFUSERS_COMMIT}
python3 -u setup.py install
rm -rf ./.git
pip install -r /content/diffusers/examples/dreambooth/requirements.txt
EOF

WORKDIR /

COPY ./entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/entrypoint.sh"]
CMD python3 --version
