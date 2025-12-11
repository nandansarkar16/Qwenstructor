#!/bin/bash
echo "Setting up environment for custom-autodan..."
echo ""

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install torch transformers>=4.35.0 numpy accelerate
pip install deepspeed 2>/dev/null || echo "Note: DeepSpeed installation may require CUDA setup"

echo ""
echo "Setup complete! Activate with: source venv/bin/activate"
echo "Then run: python example_usage.py"





