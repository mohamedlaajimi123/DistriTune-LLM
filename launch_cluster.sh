#!/bin/bash

# 1. Source a .env file ONLY if it exists locally on the machine
if [ -f .env ]; then
    echo "⚙️  Loading local cluster overrides from .env file..."
    export $(cat .env | xargs)
fi

# 2. Use the .env value if present, otherwise fall back to safe defaults
# This syntax means: "Use the existing variable, or default to this number"
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-29500}

echo "🚀 Launching DistriTune-LLM Training Cluster..."
echo "Simulating $NUM_GPUS_PER_NODE compute workers on port $MASTER_PORT..."

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    src/training/distributed_train.py

echo "✅ Cluster execution finalized safely."