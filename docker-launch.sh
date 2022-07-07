#!/bin/bash

docker run -it --gpus=all \
    -v /home/wei/:/home/penghuiwei/MyWorkspace \
    --name cu1.5.1-cudnn-devel-u20.04-penghuiwei \
    nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

