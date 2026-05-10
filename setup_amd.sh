#!/bin/bash

# Lumi — AMD MI300X Setup Script
# This script automates Phase 2: Environment Setup

echo "🚀 Starting Lumi Environment Setup on AMD MI300X..."

# 1. Install uv for fast dependency management
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
else
    echo "✅ uv is already installed."
fi

# 2. Update and Install System Dependencies
echo "🔄 Updating system packages..."
apt-get update && apt-get install -y git git-lfs

# 3. Open Firewall for vLLM
echo "🛡 Opening port 8000 for Gradio connectivity..."
# Note: ufw might need sudo if not root, but on AMD droplets you are root
ufw allow 8000

# 4. Setup Project
echo "📂 Setting up project directory..."
if [ ! -d ".git" ]; then
    # Replace with your actual repo URL
    git clone https://github.com/YUGOROU/amd-voice-sft.git
    cd amd-voice-sft
fi

# 5. Install Python Dependencies with ROCm Support
echo "🐍 Installing Python dependencies (this may take a few minutes)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
uv pip install transformers peft trl datasets huggingface_hub openai chromadb sentence-transformers python-dotenv

# 6. Set ROCm Overrides
echo "⚙️ Setting ROCm environment variables..."
export HSA_OVERRIDE_GFX_VERSION=9.4.2
echo "export HSA_OVERRIDE_GFX_VERSION=9.4.2" >> ~/.bashrc

echo "✅ Phase 2 Complete!"
echo "👉 You are now ready for Phase 3 (Training). Run: python training/train_sft.py"
