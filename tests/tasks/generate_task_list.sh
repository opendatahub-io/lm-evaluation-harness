#!/bin/bash

# Script to install lm-evaluation-harness from ODH fork and generate task list CSV
set -e

# Set default branch if not provided
BRANCH=${1:main}


# Checkout branch
git checkout $BRANCH

echo "Installing lm-evaluation-harness from OpenDataHub fork (branch: $BRANCH)..."
pip install -e .

echo "Generating task list with Python..."
python3 extract_tasks.py

echo "Script completed successfully!"
