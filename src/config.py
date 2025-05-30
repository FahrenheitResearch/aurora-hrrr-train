#!/usr/bin/env python3
"""
Configuration settings for Aurora HRRR native resolution training on H100
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class ModelConfig:
    """Aurora model configuration"""
    # Native Aurora settings (compatible dimensions)
    embed_dim: int = 1024
    encoder_depths: Tuple[int, ...] = (6, 10, 8)
    encoder_num_heads: Tuple[int, ...] = (16, 16, 16)  # Fixed: all divisible by embed_dim
    decoder_depths: Tuple[int, ...] = (8, 10, 6)
    decoder_num_heads: Tuple[int, ...] = (16, 16, 16)  # Fixed: all divisible by embed_dim
    patch_size: int = 4
    latent_levels: int = 13
    
    # Variables
    surf_vars: Tuple[str, ...] = ("2t", "10u", "10v", "msl", "cape", "cin", "refc", "hlcy", "mxuphl")
    static_vars: Tuple[str, ...] = ("lsm", "z", "slt")
    atmos_vars: Tuple[str, ...] = ("z", "u", "v", "t", "q")
    
    # Training optimizations
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    compile_model: bool = True

@dataclass
class DataConfig:
    """HRRR data configuration"""
    # Native HRRR resolution
    native_height: int = 1059
    native_width: int = 1799
    total_pixels: int = 1_904_241
    
    # Grid cropping (must be divisible by patch_size=4)
    crop_height: int = 1056  # 1059 -> 1056 (divisible by 4)
    crop_width: int = 1796   # 1799 -> 1796 (divisible by 4)
    
    # Pressure levels (13 levels to match Aurora)
    pressure_levels: List[int] = None
    
    # HRRR GRIB variable mappings
    hrrr_surf_vars: dict = None
    hrrr_atmos_vars: dict = None
    
    def __post_init__(self):
        if self.pressure_levels is None:
            self.pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        
        if self.hrrr_surf_vars is None:
            self.hrrr_surf_vars = {
                "2t": {"level_type": "heightAboveGround", "level": 2, "param": "2t"},
                "10u": {"level_type": "heightAboveGround", "level": 10, "param": "10u"},
                "10v": {"level_type": "heightAboveGround", "level": 10, "param": "10v"},
                "msl": {"level_type": "meanSea", "level": 0, "param": "mslma"},
                "cape": {"level_type": "surface", "level": 0, "param": "cape"},
                "cin": {"level_type": "surface", "level": 0, "param": "cin"},
                "refc": {"level_type": "atmosphere", "level": 0, "param": "refc"},
                "hlcy": {"level_type": "heightAboveGroundLayer", "top_level": 3000, "param": "hlcy"},
                "mxuphl": {"level_type": "heightAboveGroundLayer", "top_level": 2000, "param": "unknown", "step_type": "max"}
            }
        
        if self.hrrr_atmos_vars is None:
            self.hrrr_atmos_vars = {
                "t": {"param": "t"},
                "u": {"param": "u"},
                "v": {"param": "v"},
                "q": {"param": "q"},
                "z": {"param": "gh"}
            }

@dataclass
class TrainingConfig:
    """Training configuration for H100"""
    # H100 optimized settings
    batch_size: int = 1  # Native resolution requires batch_size=1
    accumulation_steps: int = 4  # Effective batch size = 4
    max_epochs: int = 10
    
    # Learning rate
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_steps: int = 100
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip_norm: float = 1.0
    
    # Memory optimization
    cpu_offload: bool = False  # H100 has enough VRAM
    pin_memory: bool = True
    num_workers: int = 4
    
    # Checkpointing
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    
    # Monitoring
    log_every_n_steps: int = 10
    eval_every_n_epochs: int = 1

@dataclass
class SystemConfig:
    """System and hardware configuration"""
    # Paths
    data_dir: str = "hrrr_data"
    model_dir: str = "models"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    # Hardware
    device: str = "cuda"
    precision: str = "bf16"  # H100 supports bfloat16
    num_gpus: int = 1
    
    # Memory limits
    max_memory_gb: int = 75  # Leave 5GB buffer on H100-80GB
    
    # Data loading
    prefetch_factor: int = 2
    persistent_workers: bool = True

# Global configuration
CONFIG = {
    "model": ModelConfig(),
    "data": DataConfig(),
    "training": TrainingConfig(),
    "system": SystemConfig()
}

def get_config():
    """Get complete configuration"""
    return CONFIG

def update_config_from_args(args):
    """Update configuration from command line arguments"""
    if hasattr(args, 'batch_size') and args.batch_size:
        CONFIG["training"].batch_size = args.batch_size
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        CONFIG["training"].learning_rate = args.learning_rate
    
    if hasattr(args, 'data_dir') and args.data_dir:
        CONFIG["system"].data_dir = args.data_dir
    
    if hasattr(args, 'model_dir') and args.model_dir:
        CONFIG["system"].model_dir = args.model_dir
    
    return CONFIG

def print_config():
    """Print current configuration"""
    print("🔧 Aurora HRRR Configuration:")
    print(f"  Model: {CONFIG['model'].embed_dim}D, {sum(CONFIG['model'].encoder_depths)}+{sum(CONFIG['model'].decoder_depths)} layers")
    print(f"  Grid: {CONFIG['data'].crop_height}×{CONFIG['data'].crop_width} = {CONFIG['data'].crop_height * CONFIG['data'].crop_width:,} pixels")
    print(f"  Variables: {len(CONFIG['model'].surf_vars)} surface + {len(CONFIG['model'].atmos_vars)} atmospheric")
    print(f"  Batch: {CONFIG['training'].batch_size} × {CONFIG['training'].accumulation_steps} (effective)")
    print(f"  Memory: {CONFIG['system'].max_memory_gb}GB limit, {CONFIG['system'].precision} precision")
    print(f"  Device: {CONFIG['system'].device}")

if __name__ == "__main__":
    print_config()