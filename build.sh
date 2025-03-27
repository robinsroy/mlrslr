#!/usr/bin/env bash
# exit on error
set -o errexit

# Show Python version for debugging
python --version

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Make models directory
mkdir -p models

# Train models
python train_models.py

echo "Build completed successfully!" 