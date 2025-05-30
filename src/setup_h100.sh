#!/bin/bash
# Aurora HRRR H100 Setup Script

set -e

echo "🚀 Setting up Aurora HRRR training environment for H100..."

# Check if running on H100
nvidia-smi | grep -q "H100" || echo "⚠️  Warning: H100 not detected. This setup is optimized for H100."

# Create conda environment
echo "📦 Creating conda environment..."
conda create -n aurora_h100 python=3.10 -y
source activate aurora_h100

# Install PyTorch with CUDA 11.8
echo "🔥 Installing PyTorch with CUDA 11.8..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install eccodes (required for cfgrib)
echo "🌐 Installing eccodes..."
conda install -c conda-forge eccodes -y

# Install Python requirements
echo "📋 Installing Python packages..."
pip install -r requirements.txt

# Install Microsoft Aurora
echo "🌌 Installing Microsoft Aurora..."
pip install microsoft-aurora

# Verify CUDA availability
echo "✅ Verifying CUDA setup..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test Aurora import
echo "🔬 Testing Aurora import..."
python -c "
try:
    from aurora import AuroraPretrained
    from aurora.normalisation import locations, scales
    from aurora import Batch, Metadata
    print('✅ Aurora imported successfully')
    print('Available classes:', [x for x in dir(__import__('aurora')) if 'Aurora' in x])
except ImportError as e:
    print(f'❌ Aurora import failed: {e}')
    print('Available in aurora package:', dir(__import__('aurora')))
    exit(1)
"

# Create directories
echo "📁 Creating directories..."
mkdir -p hrrr_data
mkdir -p models
mkdir -p results
mkdir -p logs

# Set up logging
echo "📊 Setting up logging..."
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Download sample data for testing
echo "🌦️  Downloading sample HRRR data for testing..."
python download_hrrr_data.py --start-date 2023-07-15 --end-date 2023-07-15 --output-dir hrrr_data --test-run

echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "1. Activate environment: conda activate aurora_h100"
echo "2. Download data: python download_hrrr_data.py --start-date 2023-07-15 --end-date 2023-07-17"
echo "3. Train model: python train_aurora_native.py"
echo "4. Run inference: python inference_native.py"
echo ""
echo "💡 Monitor GPU usage: watch -n 1 nvidia-smi"
echo "💡 Check logs: tail -f logs/training.log"