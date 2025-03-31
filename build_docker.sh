#!/bin/bash

# Build the Docker image    (option: --no-cache)  (option: --build-arg CACHE_BUST=$(date +%s))  --network=host
docker build \
    --build-arg CACHE_BUST=$(date +%s) \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t virconv-user .