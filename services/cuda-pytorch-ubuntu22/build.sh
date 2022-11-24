#!bin/bash -xe

docker build --pull -t brandnewx/ubuntu22:cuda-pytorch . && docker push brandnewx/ubuntu22:cuda-pytorch
[[ $? != 0 ]] && echo "Error!" && return 13
