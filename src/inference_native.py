#!/usr/bin/env python3
"""
Inference and visualization for native resolution Aurora model
Generate high-resolution weather predictions and convective parameter analysis
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import time

from config import get_config
from utils import setup_logging, Timer, GPUMonitor, denormalize_data, get_normalization_stats
from hrrr_dataset import HRRRNativeDataset, create_dataloader
from aurora_native import create_aurora_native

class AuroraNativeInference:
    """Inference engine for native resolution Aurora model"""
    
    def __init__(self, model_path: str, data_dir: str):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Setup monitoring
        self.gpu_monitor = GPUMonitor()
        
        # Load model
        self.setup_model()
        
        # Setup data
        self.setup_data()
    
    def setup_model(self):
        """Load trained Aurora model"""
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.logger.info(f"🌌 Loading Aurora native model from {self.model_path}")
        
        # Create model
        self.model = create_aurora_native(
            pretrained=False,
            device=self.config["system"].device
        )
        
        # Load checkpoint
        checkpoint = self.model.load_checkpoint(str(self.model_path), strict=False)
        
        self.model.eval()
        
        self.logger.info("✅ Model loaded successfully")
        self.gpu_monitor.log_memory_usage("After model loading")
    
    def setup_data(self):
        """Setup data loader"""
        
        self.logger.info(f"📊 Setting up HRRR dataset from {self.data_dir}")
        
        self.dataset = HRRRNativeDataset(
            data_dir=str(self.data_dir),
            normalize=True,
            cache_static=True
        )
        
        self.dataloader = create_dataloader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,  # Single process for inference
            pin_memory=True
        )
        
        self.logger.info(f"📈 Dataset ready: {len(self.dataset)} samples")
    
    def run_inference(self, sample_idx: int = 0) -> Dict:
        """Run inference on a single sample"""
        
        if sample_idx >= len(self.dataset):
            raise ValueError(f"Sample index {sample_idx} out of range (dataset size: {len(self.dataset)})")
        
        self.logger.info(f"🔮 Running inference on sample {sample_idx}")
        
        # Get sample
        sample = self.dataset[sample_idx]
        
        # Move to device
        self.move_batch_to_device(sample)
        
        # Run inference
        with torch.no_grad():
            with Timer("Inference"):
                predictions = self.model(sample)
        
        # Denormalize predictions and targets
        denormalized_predictions = self.denormalize_batch(predictions)
        denormalized_targets = self.denormalize_batch(sample)
        
        self.gpu_monitor.log_memory_usage("After inference")
        
        return {
            "predictions": denormalized_predictions,
            "targets": denormalized_targets,
            "metadata": sample.metadata,
            "sample_idx": sample_idx
        }
    
    def move_batch_to_device(self, batch):
        """Move batch to GPU"""
        device = self.config["system"].device
        
        # Move surface variables
        for key in batch.surf_vars:
            batch.surf_vars[key] = batch.surf_vars[key].to(device, non_blocking=True)
        
        # Move atmospheric variables
        for key in batch.atmos_vars:
            batch.atmos_vars[key] = batch.atmos_vars[key].to(device, non_blocking=True)
        
        # Move static variables
        for key in batch.static_vars:
            batch.static_vars[key] = batch.static_vars[key].to(device, non_blocking=True)
        
        # Move metadata
        for attr in ['lat', 'lon', 'time', 'atmos_levels']:
            tensor = getattr(batch.metadata, attr)
            setattr(batch.metadata, attr, tensor.to(device, non_blocking=True))
    
    def denormalize_batch(self, batch):
        """Denormalize a batch using HRRR statistics"""
        
        denormalized_batch = type(batch)(
            surf_vars={},
            atmos_vars=batch.atmos_vars.copy(),
            static_vars=batch.static_vars.copy(),
            metadata=batch.metadata
        )
        
        # Denormalize surface variables
        for var_name, tensor in batch.surf_vars.items():
            mean, std = get_normalization_stats(var_name)
            denormalized_data = tensor.cpu().numpy() * std + mean
            denormalized_batch.surf_vars[var_name] = torch.tensor(denormalized_data)
        
        return denormalized_batch
    
    def create_visualization(self, results: Dict, output_dir: str) -> List[Path]:
        """Create comprehensive visualizations"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        predictions = results["predictions"]
        targets = results["targets"]
        metadata = results["metadata"]
        sample_idx = results["sample_idx"]
        
        generated_files = []
        
        # Get coordinates
        lat = metadata.lat.cpu().numpy()
        lon = metadata.lon.cpu().numpy()
        
        # Variable information for plotting
        var_info = {
            "2t": {"name": "2m Temperature", "unit": "K", "cmap": "RdYlBu_r", "vmin": 250, "vmax": 320},
            "10u": {"name": "10m U Wind", "unit": "m/s", "cmap": "RdBu_r", "vmin": -20, "vmax": 20},
            "10v": {"name": "10m V Wind", "unit": "m/s", "cmap": "RdBu_r", "vmin": -20, "vmax": 20},
            "msl": {"name": "Mean Sea Level Pressure", "unit": "hPa", "cmap": "viridis", "vmin": 980, "vmax": 1040, "scale": 0.01},
            "cape": {"name": "CAPE", "unit": "J/kg", "cmap": "plasma", "vmin": 0, "vmax": 3000},
            "cin": {"name": "CIN", "unit": "J/kg", "cmap": "plasma_r", "vmin": -200, "vmax": 0},
            "refc": {"name": "Composite Reflectivity", "unit": "dBZ", "cmap": "turbo", "vmin": -10, "vmax": 70},
            "hlcy": {"name": "0-3km Helicity", "unit": "m²/s²", "cmap": "RdBu_r", "vmin": -500, "vmax": 500},
            "mxuphl": {"name": "Max Updraft Helicity", "unit": "m²/s²", "cmap": "hot", "vmin": 0, "vmax": 200}
        }
        
        # Create plots for each surface variable
        for var_name in self.config["model"].surf_vars:
            if var_name in predictions.surf_vars and var_name in targets.surf_vars:
                
                pred_data = predictions.surf_vars[var_name].numpy()
                target_data = targets.surf_vars[var_name].numpy()
                
                # Apply scaling if needed
                info = var_info[var_name]
                if "scale" in info:
                    pred_data = pred_data * info["scale"]
                    target_data = target_data * info["scale"]
                
                # Create comparison plot
                fig = plt.figure(figsize=(20, 12))
                
                # Target
                ax1 = plt.subplot(2, 3, 1, projection=ccrs.PlateCarree())
                self.plot_weather_map(ax1, target_data, lat, lon, info, f"Target: {info['name']}")
                
                # Prediction
                ax2 = plt.subplot(2, 3, 2, projection=ccrs.PlateCarree())
                self.plot_weather_map(ax2, pred_data, lat, lon, info, f"Prediction: {info['name']}")
                
                # Difference
                diff_data = pred_data - target_data
                diff_info = info.copy()
                diff_info["cmap"] = "RdBu_r"
                diff_info["vmin"] = -np.abs(diff_data).max()
                diff_info["vmax"] = np.abs(diff_data).max()
                
                ax3 = plt.subplot(2, 3, 3, projection=ccrs.PlateCarree())
                self.plot_weather_map(ax3, diff_data, lat, lon, diff_info, f"Difference: {info['name']}")
                
                # Statistics plots
                ax4 = plt.subplot(2, 3, 4)
                self.plot_scatter_comparison(ax4, target_data, pred_data, info)
                
                ax5 = plt.subplot(2, 3, 5)
                self.plot_histograms(ax5, target_data, pred_data, info)
                
                ax6 = plt.subplot(2, 3, 6)
                self.plot_error_statistics(ax6, target_data, pred_data, diff_data, info)
                
                plt.suptitle(f"Aurora Native: {info['name']} (Sample {sample_idx})", fontsize=16)
                plt.tight_layout()
                
                # Save plot
                output_file = output_dir / f"sample_{sample_idx:03d}_{var_name}_{info['name'].replace(' ', '_')}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                generated_files.append(output_file)
                self.logger.info(f"✅ Saved: {output_file.name}")
        
        # Create summary plot
        summary_file = self.create_summary_plot(results, output_dir)
        generated_files.append(summary_file)
        
        return generated_files
    
    def plot_weather_map(self, ax, data, lat, lon, info, title):
        """Plot weather data on map"""
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3)
        
        # Plot data
        im = ax.contourf(
            lon, lat, data,
            levels=50,
            cmap=info["cmap"],
            vmin=info.get("vmin"),
            vmax=info.get("vmax"),
            transform=ccrs.PlateCarree()
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
        cbar.set_label(f"{info['name']} ({info['unit']})")
        
        ax.set_title(title)
        ax.set_global()
        
        # Set extent to CONUS
        ax.set_extent([-130, -60, 20, 55], ccrs.PlateCarree())
    
    def plot_scatter_comparison(self, ax, target, pred, info):
        """Plot target vs prediction scatter"""
        
        target_flat = target.flatten()
        pred_flat = pred.flatten()
        
        # Sample points for performance
        if len(target_flat) > 10000:
            indices = np.random.choice(len(target_flat), 10000, replace=False)
            target_flat = target_flat[indices]
            pred_flat = pred_flat[indices]
        
        ax.scatter(target_flat, pred_flat, alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(target_flat.min(), pred_flat.min())
        max_val = max(target_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        
        # Correlation
        corr = np.corrcoef(target_flat, pred_flat)[0, 1]
        ax.text(0.05, 0.95, f'R = {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel(f'Target {info["name"]} ({info["unit"]})')
        ax.set_ylabel(f'Predicted {info["name"]} ({info["unit"]})')
        ax.set_title('Target vs Prediction')
        ax.legend()
    
    def plot_histograms(self, ax, target, pred, info):
        """Plot histograms of target and prediction"""
        
        target_flat = target.flatten()
        pred_flat = pred.flatten()
        
        ax.hist(target_flat, bins=50, alpha=0.7, label='Target', density=True)
        ax.hist(pred_flat, bins=50, alpha=0.7, label='Prediction', density=True)
        
        ax.set_xlabel(f'{info["name"]} ({info["unit"]})')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison')
        ax.legend()
    
    def plot_error_statistics(self, ax, target, pred, diff, info):
        """Plot error statistics"""
        
        # Calculate metrics
        rmse = np.sqrt(np.mean(diff**2))
        mae = np.mean(np.abs(diff))
        bias = np.mean(diff)
        
        target_std = np.std(target)
        pred_std = np.std(pred)
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'Bias': bias,
            'Target Std': target_std,
            'Pred Std': pred_std
        }
        
        # Bar plot of metrics
        bars = ax.bar(range(len(metrics)), list(metrics.values()))
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics.keys()), rotation=45)
        ax.set_ylabel(f'{info["unit"]}')
        ax.set_title('Error Statistics')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def create_summary_plot(self, results: Dict, output_dir: Path) -> Path:
        """Create summary plot with all convective variables"""
        
        predictions = results["predictions"]
        targets = results["targets"]
        sample_idx = results["sample_idx"]
        
        # Focus on convective variables
        convective_vars = ["cape", "cin", "refc", "hlcy", "mxuphl"]
        
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        
        for i, var_name in enumerate(convective_vars):
            if var_name in predictions.surf_vars:
                
                pred_data = predictions.surf_vars[var_name].numpy()
                target_data = targets.surf_vars[var_name].numpy()
                
                # Target
                ax1 = axes[0, i]
                im1 = ax1.imshow(target_data, cmap='viridis', aspect='auto')
                ax1.set_title(f'Target: {var_name.upper()}')
                plt.colorbar(im1, ax=ax1)
                
                # Prediction
                ax2 = axes[1, i]
                im2 = ax2.imshow(pred_data, cmap='viridis', aspect='auto')
                ax2.set_title(f'Prediction: {var_name.upper()}')
                plt.colorbar(im2, ax=ax2)
        
        plt.suptitle(f'Aurora Native: Convective Parameters Summary (Sample {sample_idx})', fontsize=16)
        plt.tight_layout()
        
        output_file = output_dir / f"sample_{sample_idx:03d}_convective_summary.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_file

def main():
    parser = argparse.ArgumentParser(description="Aurora native resolution inference")
    
    parser.add_argument("--model-path", required=True, help="Path to trained model")
    parser.add_argument("--data-dir", required=True, help="HRRR data directory")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--sample-indices", nargs="+", type=int, default=[0], 
                       help="Sample indices to process")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        logger.info(f"Using GPU: {gpu_name}")
    else:
        logger.warning("CUDA not available, using CPU")
    
    try:
        # Create inference engine
        inference = AuroraNativeInference(args.model_path, args.data_dir)
        
        # Process samples
        for sample_idx in args.sample_indices:
            logger.info(f"\n🔮 Processing sample {sample_idx}")
            
            # Run inference
            results = inference.run_inference(sample_idx)
            
            # Create visualizations
            output_dir = Path(args.output_dir) / f"sample_{sample_idx:03d}"
            generated_files = inference.create_visualization(results, output_dir)
            
            logger.info(f"✅ Generated {len(generated_files)} visualization files")
            
            # Print statistics
            predictions = results["predictions"]
            targets = results["targets"]
            
            print(f"\n📊 Sample {sample_idx} Statistics:")
            for var_name in inference.config["model"].surf_vars:
                if var_name in predictions.surf_vars:
                    pred = predictions.surf_vars[var_name].numpy()
                    target = targets.surf_vars[var_name].numpy()
                    
                    rmse = np.sqrt(np.mean((pred - target)**2))
                    corr = np.corrcoef(pred.flatten(), target.flatten())[0, 1]
                    
                    print(f"  {var_name}: RMSE={rmse:.3f}, Correlation={corr:.3f}")
        
        logger.info(f"\n🎉 Inference completed! Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise

if __name__ == "__main__":
    main()