# Aurora HRRR Native Resolution Training for H100

Complete package for training Microsoft Aurora on HRRR data at native resolution on NVIDIA H100.

## 🚀 Quick Start

```bash
# 1. Setup environment
bash setup_h100.sh

# 2. Download HRRR data
python download_hrrr_data.py --start-date 2023-07-15 --end-date 2023-07-17 --output-dir hrrr_data

# 3. Train native resolution model
python train_aurora_native.py --data-dir hrrr_data --output-dir models --batch-size 1

# 4. Run inference and visualization
python inference_native.py --model-path models/aurora_hrrr_native.pt --data-dir hrrr_data --output-dir results
```

## 📋 System Requirements

- **GPU**: NVIDIA H100 (80GB VRAM recommended)
- **RAM**: 128GB+ system memory
- **Storage**: 500GB+ for HRRR data and models
- **Python**: 3.9+
- **CUDA**: 11.8+

## 🌦️ Features

- **Native Resolution**: Full 1059×1799 HRRR grid (1.9M pixels)
- **All Convective Variables**: CAPE, CIN, Radar Reflectivity, Helicity, Max Updraft Helicity
- **Full Aurora Model**: 1.3B parameters with proper fine-tuning
- **Memory Optimized**: Gradient checkpointing, mixed precision, data streaming
- **Production Ready**: Robust error handling, logging, monitoring

## 📁 File Structure

```
src/
├── README.md                 # This file
├── setup_h100.sh            # Environment setup script
├── requirements.txt          # Python dependencies
├── download_hrrr_data.py     # HRRR data downloader
├── hrrr_dataset.py          # Native resolution dataset
├── aurora_native.py         # Full Aurora model wrapper
├── train_aurora_native.py   # H100 training script
├── inference_native.py     # Inference and visualization
├── utils.py                 # Utility functions
└── config.py               # Configuration settings
```

## 🔧 Configuration

Edit `config.py` to customize:
- Grid resolution (default: native 1059×1799)
- Batch size (default: 1 for H100)
- Learning rate and training parameters
- Data paths and model settings

## 📊 Model Specifications

- **Parameters**: 1.3B (full Microsoft Aurora)
- **Grid**: 1059×1799 (1,904,241 pixels)
- **Variables**: 9 surface + 5 atmospheric + 3 static
- **Memory**: ~75GB VRAM (H100 80GB)
- **Training**: Mixed precision FP16/BF16

## 🌩️ Convective Variables

All HRRR convective parameters included:
- **CAPE**: Convective Available Potential Energy
- **CIN**: Convective Inhibition  
- **REFC**: Composite Radar Reflectivity
- **HLCY**: Storm-Relative Helicity (0-3km)
- **MXUPHL**: Maximum Updraft Helicity (2-5km)

## 💾 Data Requirements

- **HRRR Surface**: ~2GB per day
- **HRRR Pressure**: ~8GB per day
- **Total**: ~10GB per day of data
- **Recommended**: 1 week = ~70GB

## ⚡ Performance

Expected on H100:
- **Training**: ~10-15 minutes per epoch (100 samples)
- **Inference**: ~30 seconds per forecast
- **Memory**: 70-75GB VRAM usage
- **Throughput**: 1-2 samples/minute

## 🛠️ Troubleshooting

### Out of Memory
- Reduce batch size to 1
- Enable gradient checkpointing
- Use CPU offloading for optimizer

### Slow Training
- Check data loading (use SSD storage)
- Monitor GPU utilization
- Verify mixed precision is enabled

### Poor Convergence
- Adjust learning rate (start with 1e-5)
- Check data normalization
- Verify convective variable extraction

## 📈 Monitoring

Training logs include:
- Loss curves (total, per-variable)
- Memory usage (GPU/CPU)
- Data loading times
- Gradient norms
- Learning rate schedule

## 🚀 Advanced Usage

### Multi-GPU Training
```bash
torchrun --nproc_per_node=8 train_aurora_native.py --distributed
```

### Custom Data Range
```bash
python download_hrrr_data.py --start-date 2023-06-01 --end-date 2023-08-31 --variables cape,cin,refc,hlcy
```

### Inference Only
```bash
python inference_native.py --model-path pretrained/aurora_hrrr.pt --sample-count 10
```

## 📄 License

Based on Microsoft Aurora (MIT) and NOAA HRRR data (public domain).

## 🙏 Acknowledgments

- Microsoft Aurora team
- NOAA HRRR dataset
- PyTorch and xarray communities