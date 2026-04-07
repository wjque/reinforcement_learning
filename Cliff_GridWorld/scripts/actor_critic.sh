#!/bin/bash

# Please run under directory: Cliff_GridWorld
python train.py \
    --algo "actor_critic" \
    --device "cuda" \
    --seed 42 \
    --episodes 2000