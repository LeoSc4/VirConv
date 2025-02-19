#!/bin/bash
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

is_image_build() {
    if [ "$(docker images -q virconv-pytorch 2> /dev/null)" == "" ]; then
        echo -e "${RED}Image not found. Have you built the image?${NC}"
        echo -e "${BLUE}Try running: ./build_docker.sh${NC}"
        return 1
    else
        echo -e "${GREEN}Image found.${NC}"
        return 0
    fi
}

run_docker() {
    is_image_build
    if [ $? -eq 0 ]; then
        echo -e "${YELLOW}Running docker...${NC}"
        xhost +local:root
        docker run \
            --runtime nvidia \
            --name virconv-pytorch \
            -it \
            --net host \
            --gpus all \
            --rm \
            -v "/home/leo/workspace/Docker_tests/VirConv:/workspace" \
            virconv-pytorch
            
    fi
}

run_docker
