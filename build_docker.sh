#!/bin/bash

# Build the Docker image    (option: --no-cache)  (option: --build-arg CACHE_BUST=$(date +%s))
docker build --build-arg CACHE_BUST=$(date +%s) -t virconv-pytorch .