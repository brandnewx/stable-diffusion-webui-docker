FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu22.04 as cuda-pytorch-ubuntu22
USER root
WORKDIR /
SHELL ["/bin/bash", "-ceuxo", "pipefail"]

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1
ENV TZ=Asia/Phnom_Penh

ARG XFORMERS_SHA=103e863db94f712a96c34fc8e78cfd58a40adeee

# set to compile for Nvidia T, A, RTX (see https://developer.nvidia.com/cuda-gpus#compute)
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"

RUN apt-get update -q && apt-get install -y --no-install-recommends \
    linux-headers-$(uname -r) \
    build-essential \
    ca-certificates \
    apt-transport-https \
    gnupg gnupg2 \
    lsb-release \
    software-properties-common \
    wget curl \
    zlib1g \
    git git-lfs \
    moreutils \
    cifs-utils \
    dirmngr \
    expat \
    libgomp1 \
    python3.10 python3-pip \
    zip unzip unrar \
    xvfb \
    imagemagick \
    cifs-utils \
    ruby \
    file \
    cron \
    rsyslog \
    rsync \
    rclone \
    youtube-dl \
    streamlink \
    ffmpeg \
    ncdu mc && \
    apt-get clean

RUN apt-get update -qq && apt-get install -yq apt-transport-https ca-certificates curl gnupg gnupg2 lsb-release software-properties-common \
    wget zlib1g git moreutils && apt-get clean
    
RUN apt-get update -qq && apt-get install -yq python3.10 python3-pip && apt-get clean

RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# to fix out of index error in torch/utils/cpp_extension.py
ENV FORCE_CUDA=1
ENV USE_CUDA=1

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

# Compile and install xformers
RUN git clone https://github.com/facebookresearch/xformers.git && \
  cd xformers && \
  git fetch && \
  git submodule update --init --recursive && \
  git reset --hard ${XFORMERS_SHA} && \
  pip install -r requirements.txt && \
  pip install wheel==0.38.4 ninja==1.11.1 triton==2.0.0.dev20221117 && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH && \
  cd /xformers && python3 setup.py build && \
  cd /xformers && python3 setup.py bdist_wheel && \
  pip install /xformers/dist/*.whl && rm -rf /xformers
