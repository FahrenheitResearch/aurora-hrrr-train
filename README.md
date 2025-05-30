# Aurora HRRR Fine-tuning for Convective Modeling

This repository implements **Strategy 1: Basic Variable Extension** from the Aurora HRRR fine-tuning guide, enabling Aurora to predict convective parameters like radar reflectivity, CAPE, CIN, and severe weather indicators using HRRR data.

## 🚀 Quick Start

### 1. Installation
```bash
# Clone this repository
git clone <repository-url>
cd weather-model-hrrr-tune

# Install dependencies
pip install -r requirements.txt

# Install Aurora (requires separate installation)
pip install git+https://github.com/microsoft/aurora.git
```

### 2. Demo with Synthetic Data
```bash
python example_usage.py --demo
```

### 3. Download HRRR Data
```bash
python hrrr_downloader.py \
    --start-date 2023-06-01 \
    --end-date 2023-06-30 \
    --output-dir ./hrrr_data \
    --training-dataset
```

### 4. Train the Model
```bash
python train_aurora_convective.py \
    --data-dir ./hrrr_data \
    --start-date 2023-06-01 \
    --end-date 2023-08-31 \
    --epochs 20 \
    --checkpoint-dir ./checkpoints
```

## 📋 Components

### Core Implementation Files

| File | Description |
|------|-------------|
| `hrrr_downloader.py` | Downloads HRRR GRIB2 files from NOAA S3 bucket |
| `hrrr_to_aurora.py` | Converts HRRR data to Aurora batch format |
| `aurora_convective.py` | Extended Aurora model with convective variables |
| `train_aurora_convective.py` | Training script with proper normalization |
| `convective_metrics.py` | Comprehensive evaluation metrics |
| `example_usage.py` | Complete usage examples and demos |

### Documentation Files

| File | Description |
|------|-------------|
| `AURORA_HRRR_FINETUNING_OPTIONS.md` | Comprehensive guide with 6 implementation strategies |
| `HRRR_DATA_CHEATSHEET.md` | HRRR data extraction reference |
| `AURORA.MD` | Aurora fine-tuning documentation |

## 🌩️ Convective Variables Added

The extended Aurora model includes these convective parameters from HRRR:

| Variable | Description | Units | Use Case |
|----------|-------------|-------|----------|
| `cape` | Convective Available Potential Energy | J/kg | Convective instability |
| `cin` | Convective Inhibition | J/kg | Convective suppression |
| `refc` | Composite Radar Reflectivity | dBZ | Precipitation intensity |
| `hlcy` | Storm-Relative Helicity (0-3km) | m²/s² | Rotation potential |
| `mxuphl` | Max Updraft Helicity (2-5km) | m²/s² | Supercell/tornado potential |

## 🔧 Key Features

### Smart Data Handling
- **Parallel downloading** from NOAA S3 bucket
- **Automatic coordinate conversion** (Lambert Conformal → lat/lon)
- **Pressure level interpolation** (HRRR 40 levels → Aurora 13 levels)
- **Quality validation** with meteorological sanity checks

### Advanced Training
- **Differential learning rates** for new vs. existing variables
- **Gradient stabilization** with layer normalization
- **Physics-aware loss weighting** for convective variables
- **Mixed precision training** for memory efficiency

### Comprehensive Evaluation
- **Meteorological skill scores** (CSI, ETS, POD, FAR)
- **Structure-Amplitude-Location (SAL)** analysis for spatial patterns
- **Threshold-based metrics** for severe weather events
- **Convective regime classification** (supercell potential, etc.)

## 📊 Example Metrics

### Reflectivity Skill Scores
```
Light Precipitation (20 dBZ):   CSI: 0.65, ETS: 0.45
Moderate Precipitation (35 dBZ): CSI: 0.45, ETS: 0.35
Heavy Precipitation (50 dBZ):    CSI: 0.25, ETS: 0.20
```

### CAPE/CIN Performance
```
CAPE RMSE: 850 J/kg, Correlation: 0.75
CIN RMSE: 45 J/kg, Correlation: 0.68
Convective Potential CSI: 0.55
```

## 🎯 Training Strategy

### Phase 1: Basic Extension (Weeks 1-2)
- Add 5 key convective variables
- Use existing Aurora normalization
- Train for 10-20 epochs

### Phase 2: Optimization (Weeks 3-4)
- Fine-tune normalization statistics
- Implement physics-informed loss
- Validate on diverse convective events

### Phase 3: Evaluation (Weeks 5-6)
- Test on severe weather cases
- Compare against operational forecasts
- Generate performance reports

## 💻 Hardware Requirements

### Minimum Configuration
- **GPU**: A100 80GB for gradient computation
- **Memory**: 256GB+ RAM for HRRR processing
- **Storage**: 10TB+ for dataset (1 year ≈ 50TB)

### Recommended Configuration
- **GPU**: 4x A100 80GB for parallel training
- **Memory**: 512GB+ RAM
- **Storage**: 100TB+ with high-speed I/O

## 📈 Expected Performance

### Short-term (1-2 months)
- Basic convective parameter prediction
- Improved precipitation forecasts
- Better diurnal convective cycles

### Medium-term (3-6 months)  
- Accurate storm-scale features
- Supercell environment identification
- Multi-hour convective evolution

### Long-term (6-12 months)
- Operational-quality 3km forecasts
- Reliable severe weather guidance
- Ensemble integration capability

## 🔬 Evaluation Commands

### Generate Metrics Report
```bash
python convective_metrics.py \
    --predictions ./predictions.pt \
    --targets ./targets.pt \
    --output-report ./evaluation_report.txt
```

### Visualize Performance
```bash
python visualize_results.py \
    --checkpoint ./checkpoints/best_checkpoint.pt \
    --test-data ./test_data \
    --output-dir ./plots
```

## 🚨 Important Notes

### Data Quality
- Always validate HRRR data for missing/corrupt files
- Check coordinate systems and projections
- Monitor for unrealistic values (NaN, extreme outliers)

### Training Stability
- Use gradient clipping (max_norm=1.0)
- Enable `stabilise_level_agg=True` for new variables
- Monitor gradient magnitudes for explosion

### Memory Management
- Use gradient checkpointing for large batches
- Enable automatic mixed precision (AMP)
- Consider data parallel training for multiple GPUs

## 📚 References

- [Aurora Paper](https://www.nature.com/articles/s41586-025-09005-y)
- [HRRR Documentation](https://rapidrefresh.noaa.gov/hrrr/)
- [AWS HRRR Data](https://registry.opendata.aws/noaa-hrrr/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement improvements
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Microsoft Aurora team for the foundational model
- NOAA for providing HRRR data
- Open source meteorological community

---

**Ready to predict the next supercell? Start with the demo and scale up to operational forecasting!** 🌪️