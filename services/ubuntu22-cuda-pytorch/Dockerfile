FROM nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu22.04
USER root
WORKDIR /
SHELL ["/bin/bash", "-ceuxo", "pipefail"]

ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 PIP_NO_CACHE_DIR=1
ENV TZ=Asia/Phnom_Penh

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
    dumb-init \
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
