#!/bin/bash
# Setup script for concept learning environment

# Set base directory
export BASE_DIR="/bigtemp/nkw3mr/concept-prototype-learing"

# Set cache directories to bigtemp to save home directory space
export HF_HOME="${BASE_DIR}/huggingface_cache"
export TRANSFORMERS_CACHE="${BASE_DIR}/huggingface_cache"
export HF_DATASETS_CACHE="${BASE_DIR}/huggingface_cache/datasets"
export TORCH_HOME="${BASE_DIR}/torch_cache"

# Create cache directories if they don't exist
mkdir -p "${HF_HOME}"
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${TORCH_HOME}"

# Load modules
module load gcc/11.4.0 cuda/12.8.1

# Activate conda environment

echo "Environment setup complete!"
echo "Cache directories:"
echo "  HF_HOME: ${HF_HOME}"
echo "  TRANSFORMERS_CACHE: ${TRANSFORMERS_CACHE}"
echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
echo "  TORCH_HOME: ${TORCH_HOME}"