#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Train models
python train_models.py

echo "Build completed successfully!" 