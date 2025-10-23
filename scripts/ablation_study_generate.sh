#!/bin/bash
# LLM Ablation Study - Data Generation
# Generates data with each LLM (gpt5mini, gpt41, claude)
# 
# This script generates benchmark data using different LLMs to evaluate
# how data generation quality affects model performance.

set -e

echo "=========================================="
echo "PROBE LLM Ablation Study - Data Generation"
echo "=========================================="
echo ""

# Data generation models
DATA_GEN_MODELS=("gpt5mini" "gpt41" "claude")
DATA_GEN_CONFIGS=(
    "data_generation/config.ablation_gpt5mini.yaml"
    "data_generation/config.ablation_gpt41.yaml"
    "data_generation/config.ablation_claude.yaml"
)

echo "Generating data with each LLM..."
echo ""

for i in "${!DATA_GEN_MODELS[@]}"; do
    model="${DATA_GEN_MODELS[$i]}"
    config="${DATA_GEN_CONFIGS[$i]}"
    
    echo ""
    echo "▶ Generating data with: $model"
    echo "  Config: $config"
    echo ""
    
    # Run data generation (doppler handles environment variables)
    doppler run -- python run.py --config "$config"
    
    echo "✓ Data generation complete: $model"
    echo ""
done

echo ""
echo "=========================================="
echo "Data Generation Complete!"
echo "=========================================="
echo ""
echo "Generated data saved to: generated_data/ablation_*/<timestamp>_batch/"
echo "  - inputs/  : Input files for models (*_input.json)"
echo "  - outputs/ : Ground truth labels (*_output.json)"
echo ""
echo "Next: Run ablation_study_inference.sh"
echo ""

