#!bin/bash -xe

docker build --pull -t brandnewx/ubuntu22:cuda-pytorch . && docker push brandnewx/ubuntu22:cuda-pytorch
[[ $? != 0 ]] && echo "Error!" && return 13

smi=$(nvidia-smi)
# set to compile for Nvidia T, A, RTX (see https://developer.nvidia.com/cuda-gpus#compute)
if [[ $smi == *"A100"* ]]; then
  docker build --pull --build-arg TORCH_CUDA_ARCH_LIST="8.0+PTX" -t brandnewx/ubuntu22:cuda-pytorch-xformers-a100 . && docker push brandnewx/ubuntu22:cuda-pytorch-xformers-a100
else if [[ $smi == *"T4"* ]]; then
  docker build --pull --build-arg TORCH_CUDA_ARCH_LIST="7.5+PTX" -t brandnewx/ubuntu22:cuda-pytorch-xformers-t4 . && docker push brandnewx/ubuntu22:cuda-pytorch-xformers-t4
fi
