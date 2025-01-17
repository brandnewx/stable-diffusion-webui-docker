FROM brandnewx/ubuntu22:cuda-pytorch
USER root
WORKDIR /
SHELL ["/bin/bash", "-ceuxo", "pipefail"]

# Need dev package(s) to compile
RUN apt-get update -qq && apt-get install -y python3.10-dev && apt-get clean

ARG XFORMERS_SHA=103e863db94f712a96c34fc8e78cfd58a40adeee

# set to compile for Nvidia T, A, RTX (see https://developer.nvidia.com/cuda-gpus#compute)
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"

# to fix out of index error in torch/utils/cpp_extension.py
ENV FORCE_CUDA=1 USE_CUDA=1

# Torch build env variables
ENV USE_OPENCV=1 \
  BUILD_TORCH=ON \
  CMAKE_PREFIX_PATH="/usr/bin/" \
  CUDA_BIN_PATH=/usr/local/cuda/bin \
  CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
  CUDNN_LIB_DIR=/usr/local/cuda/lib64 \
  CUDA_HOST_COMPILER=cc USE_NNPACK=1 CC=cc CXX=c++ \
  TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Compile and install xformers
RUN git clone https://github.com/facebookresearch/xformers.git && \
  cd xformers && \
  git fetch && \
  git reset --hard ${XFORMERS_SHA} && \
  git submodule update --init --recursive && \
  pip install -r requirements.txt && \
  pip install wheel==0.38.4 ninja==1.11.1 triton==2.0.0.dev20221117 && \
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH && \
  cd /xformers && python3 setup.py build && \
  cd /xformers && python3 setup.py bdist_wheel && \
  mv /xformers/dist /xformers-dist && rm -rf /xformers
  
