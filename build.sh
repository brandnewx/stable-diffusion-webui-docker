#!bin/bash -xe

echo "building brandnewx/ubuntu:cuda-pytorch..."
docker build -t brandnewx/ubuntu22:cuda-pytorch ./services/ubuntu22-cuda-pytorch/
[[ $? != 0 ]] && echo "Error!" && return 13

# set to compile xformers for different nvidia computes (see https://developer.nvidia.com/cuda-gpus#compute)
smi=$(nvidia-smi)
if [[ $smi == *"A100"* ]]; then
  echo "building brandnewx/xformers:a100 ..."
  docker build -t ubuntu22:xformers:a100 ./services/xformers/Dockerfile.a100
elif [[ $smi == *"T4"* ]]; then
  echo "building brandnewx/xformers:t4 ..."
  docker build -t brandnewx/xformers:t4 ./services/xformers/Dockerfile.t4
else
  echo "skipped building xformers as current gpu is not supported!"
fi
[[ $? != 0 ]] && echo "Error!" && return 14

echo "building automatic1111:t4"
docker build -t automatic1111:t4 ./services/AUTOMATIC1111/

[[ $? != 0 ]] && echo "Error!" && return 14
