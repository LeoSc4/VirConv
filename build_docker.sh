#!/bin/bash

# Build the Docker image    (option: --no-cache)  (option: --build-arg CACHE_BUST=$(date +%s))  --network=host
docker build --no-cache -t virconv-pytorch1131-cuda117-wandb .