#!/bin/bash

# Dynamic batch equivalent of proactive_prep.sh - CLI driven with native batch APIs
# 50% cost savings vs individual API calls

# Load environment variables from .env file
if [ -f "../../.env" ]; then
    export $(cat ../../.env | grep -v '^#' | xargs)
elif [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set. Please set it in .env file or as an environment variable."
    exit 1
fi

cd ~/Documents/proactive-baselines

echo "🎯 Running Batch Inference with OpenAI"
echo "======================================================="
echo ""
echo ""
python run_native_batch_evaluation.py \
    --models \
        MODELS.OPENAI.GPT_5 \
    --data_dir data/sept_23_1000_inputs_20250923_131956/