# syntax=docker/dockerfile:1

FROM alpine/git:2.36.2 as download
SHELL ["/bin/sh", "-ceuxo", "pipefail"]

RUN <<EOF
cat <<'EOE' > /clone.sh
mkdir -p repositories/"$1" && cd repositories/"$1" && git init && git remote add origin "$2" && git fetch origin "$3" --depth=1 && git reset --hard "$3" && rm -rf .git
EOE
EOF

RUN . /clone.sh taming-transformers https://github.com/CompVis/taming-transformers.git 24268930bf1dce879235a7fddd0b2355b84d7ea6 \
  && rm -rf data assets **/*.ipynb

RUN . /clone.sh stable-diffusion https://github.com/CompVis/stable-diffusion.git 69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc \
  && rm -rf assets data/**/*.png data/**/*.jpg data/**/*.gif

RUN . /clone.sh CodeFormer https://github.com/sczhou/CodeFormer.git c5b4593074ba6214284d6acd5f1719b6c5d739af \
  && rm -rf assets inputs

RUN . /clone.sh BLIP https://github.com/salesforce/BLIP.git 48211a1594f1321b00f14c9f7a5b4813144b2fb9
RUN . /clone.sh k-diffusion https://github.com/crowsonkb/k-diffusion.git 60e5042ca0da89c14d1dd59d73883280f8fce991
RUN . /clone.sh clip-interrogator https://github.com/pharmapsychotic/clip-interrogator 2486589f24165c8e3b303f84e9dbbea318df83e8

# FROM alpine:3 as xformers
# RUN apk add aria2
# RUN aria2c --dir / --out wheel.whl 'https://github.com/AbdBarho/stable-diffusion-webui-docker/releases/download/2.1.0/xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl'


FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu22.04 as cuda-pytorch-ubuntu22
USER root
WORKDIR /
SHELL ["/bin/bash", "-ceuxo", "pipefail"]

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1

RUN apt-get update -qq && apt-get install -yq apt-transport-https ca-certificates curl gnupg gnupg2 lsb-release software-properties-common \
    wget zlib1g git moreutils && apt-get clean
    
RUN apt-get update -qq && apt-get install -yq python3.10 python3-pip && apt-get clean

RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install CUDA libs (some libs like bitandbytes need CUDA to compile).
#ENV cudnn_version=8.5.0.96
#ENV cuda_version=cuda11.7
#ENV OS=ubuntu2204
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin && \
#  mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
#  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/3bf863cc.pub && \
#  add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /" && \
#  apt-get update && apt-get install -yq libcudnn8=${cudnn_version}-1+${cuda_version} libcudnn8-dev=${cudnn_version}-1+${cuda_version}


FROM cuda-pytorch-ubuntu22 as xformers-ubuntu22
USER root
WORKDIR /
SHELL ["/bin/bash", "-ceuxo", "pipefail"]

# to fix out of index error in torch/utils/cpp_extension.py
ENV FORCE_CUDA=1
ENV USE_CUDA=1

# set to compile for T4 and A100 (see https://developer.nvidia.com/cuda-gpus#compute)
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"

# Torch build env variables
#ENV USE_OPENCV=1
#ENV BUILD_TORCH=ON
#ENV CMAKE_PREFIX_PATH="/usr/bin/"
#ENV CUDA_BIN_PATH=/usr/local/cuda/bin
#ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/
#ENV CUDNN_LIB_DIR=/usr/local/cuda/lib64
#ENV CUDA_HOST_COMPILER=cc
#ENV USE_NNPACK=1
#ENV CC=cc
#ENV CXX=c++
#ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

ARG XFORMERS_SHA=103e863db94f712a96c34fc8e78cfd58a40adeee
RUN git clone https://github.com/facebookresearch/xformers.git && \
  cd xformers && \
  git fetch && \
  git reset --hard ${XFORMERS_SHA} && \
  git submodule update --init --recursive && \
  pip install -r requirements.txt && \
  pip install wheel ninja triton
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH
RUN cd /xformers && python3 setup.py build
RUN cd /xformers && python3 setup.py bdist_wheel


FROM cuda-pytorch-ubuntu22

USER root
WORKDIR /

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

RUN apt-get update -qq && apt-get install -yq fonts-dejavu-core rsync git jq moreutils && apt-get clean

RUN git clone https://github.com/brandnewx/stable-diffusion-webui.git && \
  cd stable-diffusion-webui && \
  # git reset --hard 284cf8f322110df0920ec022937d0ebfe9a61550 && \
  pip install -r requirements_versions.txt

COPY --from=xformers-ubuntu22 /xformers/dist/ /xformers/dist/
RUN pip install /xformers/dist/*.whl && rm -rf /xformers

ENV ROOT=/stable-diffusion-webui


COPY --from=download /git/ ${ROOT}
RUN mkdir ${ROOT}/interrogate && cp ${ROOT}/repositories/clip-interrogator/data/* ${ROOT}/interrogate
RUN pip install -r ${ROOT}/repositories/CodeFormer/requirements.txt


ARG DEEPDANBOORU="0"
RUN [[ "${DEEPDANBOORU:-0}" == "0" ]] && : || pip install tensorflow-cpu==2.10 tensorflow-io==0.27.0 git+https://github.com/KichangKim/DeepDanbooru.git@edf73df4cdaeea2cf00e9ac08bd8a9026b7a7b26#egg=deepdanbooru

RUN pip install opencv-python-headless \
  git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379 \
  git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1 \
  pyngrok

# Note: don't update the sha of previous versions because the install will take forever
# instead, update the repo state in a later step

ARG SHA=e26174a81f81ce876d037877c33795424208b090
RUN cd stable-diffusion-webui && \
  git fetch && \
# git reset --hard ${SHA} && \
  git reset --hard && \
  pip install -r requirements_versions.txt

RUN pip install opencv-python-headless triton

COPY . /docker

RUN python3 /docker/info.py ${ROOT}/modules/ui.py && \
  mv  ${ROOT}/style.css ${ROOT}/user.css && \
  sed -i 's/os.rename(tmpdir, target_dir)/shutil.move(tmpdir,target_dir)/' ${ROOT}/modules/ui_extensions.py

# Install extensions
RUN <<EOF
cat <<'EOE' > /install-extension.sh
mkdir -p ${ROOT}/extensions/"$1" && cd ${ROOT}/extensions/"$1" && git init && git remote add origin "$2" && git fetch origin "$3" --depth=1 && git reset --hard "$3" && rm -rf .git
cd ${ROOT}/extensions/"$1"
[ -f requirements.txt ] && pip install -r requirements.txt
exit 0
EOE
EOF

RUN . /install-extension.sh sd_dreambooth_extension https://github.com/d8ahazard/sd_dreambooth_extension 725d85ca5485246a430d2f31f624646d061e8104
RUN . /install-extension.sh stable-diffusion-webui-aesthetic-gradients https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients 411889ca602f20b8bb5e4d1af2b9686eab1913b1
RUN . /install-extension.sh stable-diffusion-webui-wildcards https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards 6ed81ed1eaee9145b596ab6385c38ffd8ce7b13f
RUN . /install-extension.sh sd-dynamic-prompts https://github.com/adieyal/sd-dynamic-prompts c7434e7822bdb1d2c2f117d0edbfe28fd1f63461
RUN . /install-extension.sh stable-diffusion-webui-images-browser https://github.com/yfszzx/stable-diffusion-webui-images-browser 15528557c0434f66fd863347e2ecaff72f4f6f12
# RUN . /install-extension.sh stable-diffusion-webui-aesthetic-image-scorer https://github.com/tsngo/stable-diffusion-webui-aesthetic-image-scorer e6389da7a5827e700522ff1943ee4ae2c5010089
# RUN . /install-extension.sh auto-sd-paint-ext https://github.com/Interpause/auto-sd-paint-ext 7401bfcd05f382c56cdee147e089fc0a4ca0ea34
# RUN . /install-extension.sh a1111-sd-webui-tagcomplete https://github.com/DominikDoom/a1111-sd-webui-tagcomplete 894335f1dead68cf82ae147d3e53b37abe8fb597
# RUN . /install-extension.sh ddetailer https://github.com/dustysys/ddetailer 8e58e7fec774cfe41490a5dda6458fcbbea78493
RUN . /install-extension.sh DreamArtist-sd-webui-extension https://github.com/7eu7d7/DreamArtist-sd-webui-extension 9e0d7da5b445d0342047c0b022baf114c8d30b33
RUN python3 -u ${ROOT}/webui.py --test --ui-debug-mode --listen --port 7860 --ckpt-dir ${ROOT}/models/Stable-diffusion --allow-code --medvram --xformers --force-enable-xformers --opt-split-attention --enable-insecure-extension-access --administrator

RUN pip install bitsandbytes

# WORKDIR ${ROOT}/repositories/stable-diffusion
WORKDIR ${ROOT}
ENV CLI_ARGS=""
EXPOSE 7860
ENTRYPOINT ["/docker/entrypoint.sh"]
# run, -u to not buffer stdout / stderr
CMD python3 -u ${ROOT}/webui.py --listen --port 7860 --ckpt-dir ${ROOT}/models/Stable-diffusion ${CLI_ARGS}
