#!bin/bash -xe

echo "building ubuntu:cuda-pytorch..."
docker build --pull -t brandnewx/ubuntu22:cuda-pytorch ./services/cuda-pytorch-ubuntu22/ && docker push brandnewx/ubuntu22:cuda-pytorch
[[ $? != 0 ]] && echo "Error!" && return 13

# set to compile xformers for different nvidia computes (see https://developer.nvidia.com/cuda-gpus#compute)
smi=$(nvidia-smi)
if [[ $smi == *"A100"* ]]; then
  echo "building ubuntu22:cuda-pytorch-xformers-a100..."
  docker build --pull --build-arg TORCH_CUDA_ARCH_LIST="8.0+PTX" -t brandnewx/ubuntu22:cuda-pytorch-xformers-a100 ./services/cuda-pytorch-ubuntu22/xformers/ && docker push brandnewx/ubuntu22:cuda-pytorch-xformers-a100
else if [[ $smi == *"T4"* ]]; then
  echo "building ubuntu22:cuda-pytorch-xformers-t4"
  docker build --pull --build-arg TORCH_CUDA_ARCH_LIST="7.5+PTX" -t brandnewx/ubuntu22:cuda-pytorch-xformers-t4 ./services/cuda-pytorch-ubuntu22/xformers/ && docker push brandnewx/ubuntu22:cuda-pytorch-xformers-t4
else
  echo "skipped building xformers as current gpu is not supported!"
fi
[[ $? != 0 ]] && echo "Error!" && return 14

echo "building ubuntu22:cuda-pytorch-xformers-t4"
docker build --pull -t ubuntu22:automatic1111-t4 ./services/AUTOMATIC1111/

[[ $? != 0 ]] && echo "Error!" && return 14
