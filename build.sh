#!bin/bash -xe

echo "building brandnewx/ubuntu:cuda-pytorch..."
#docker buildx build --pull -t brandnewx/ubuntu22:cuda-pytorch ./services/ubuntu22-cuda-pytorch/ && docker push brandnewx/ubuntu22:cuda-pytorch
docker buildx build -t brandnewx/ubuntu22:cuda-pytorch ./services/ubuntu22-cuda-pytorch/ 
[[ $? != 0 ]] && echo "Error!" && return 13

#docker buildx build --pull -t brandnewx/xformers ./services/xformers # && docker push brandnewx/xformers:latest
docker buildx build -t brandnewx/xformers ./services/xformers
[[ $? != 0 ]] && echo "Error!" && return 14

echo "building automatic1111"
docker buildx build -t brandnewx/automatic1111 ./services/AUTOMATIC1111/
[[ $? != 0 ]] && echo "Error!" && return 15
