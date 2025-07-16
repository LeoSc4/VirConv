#!/bin/bash

docker build --build-arg CACHE_BUST=$(date +%s) -t adtc-object_detection .
