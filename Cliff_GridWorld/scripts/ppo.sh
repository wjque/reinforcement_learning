#!/bin/bash

# Please run under directory: Cliff_GridWorld
python train.py \
    --algo "ppo" \
    --device "cuda" \
    --seed 42 \
    --episodes 500