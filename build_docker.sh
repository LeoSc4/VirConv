#!/bin/bash

# Build the Docker image
docker build --build-arg CACHE_BUST=$(date +%s) -t virconv-pytorch .
