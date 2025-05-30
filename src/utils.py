#!/usr/bin/env python3
"""
Utility functions for Aurora HRRR training on H100
"""

import os
import sys
import time
import logging
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Setup logging
def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(f"{log_dir}/training.log")
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

class GPUMonitor:
    """Monitor GPU memory and utilization"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.logger = logging.getLogger(__name__)
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory info in GB"""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
        
        allocated = torch.cuda.memory_allocated(self.device_id) / 1e9
        reserved = torch.cuda.memory_reserved(self.device_id) / 1e9
        
        # Get total memory
        props = torch.cuda.get_device_properties(self.device_id)
        total = props.total_memory / 1e9
        free = total - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total
        }
    
    def log_memory_usage(self, prefix: str = "GPU Memory"):
        """Log current memory usage"""
        info = self.get_memory_info()
        self.logger.info(
            f"{prefix}: {info['allocated']:.1f}GB allocated, "
            f"{info['reserved']:.1f}GB reserved, "
            f"{info['free']:.1f}GB free / {info['total']:.1f}GB total"
        )
    
    def check_memory_limit(self, limit_gb: float = 75.0) -> bool:
        """Check if memory usage is within limit"""
        info = self.get_memory_info()
        return info["reserved"] < limit_gb

class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_system_info(self) -> Dict[str, float]:
        """Get system resource info"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_total_gb": memory.total / 1e9,
            "memory_used_gb": memory.used / 1e9,
            "memory_percent": memory.percent
        }
    
    def log_system_usage(self):
        """Log current system usage"""
        info = self.get_system_info()
        self.logger.info(
            f"System: CPU {info['cpu_percent']:.1f}%, "
            f"RAM {info['memory_used_gb']:.1f}GB / {info['memory_total_gb']:.1f}GB "
            f"({info['memory_percent']:.1f}%)"
        )

class Timer:
    """Simple timer for performance monitoring"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.logger = logging.getLogger(__name__)
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name}: {elapsed:.2f}s")

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """Save training checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": time.time()
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, path)
    logging.getLogger(__name__).info(f"Checkpoint saved: {path}")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda"
) -> Dict:
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    logging.getLogger(__name__).info(f"Checkpoint loaded: {path}")
    return checkpoint

def calculate_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Calculate model size and parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory (4 bytes per float32 parameter)
    memory_mb = total_params * 4 / 1e6
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "memory_mb": memory_mb
    }

def validate_grid_dimensions(height: int, width: int, patch_size: int = 4) -> Tuple[int, int]:
    """Validate and adjust grid dimensions to be divisible by patch_size"""
    # Ensure divisible by patch_size
    adj_height = (height // patch_size) * patch_size
    adj_width = (width // patch_size) * patch_size
    
    if adj_height != height or adj_width != width:
        logging.getLogger(__name__).warning(
            f"Grid adjusted from {height}×{width} to {adj_height}×{adj_width} "
            f"(must be divisible by patch_size={patch_size})"
        )
    
    return adj_height, adj_width

def normalize_data(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normalize data using mean and std"""
    return (data - mean) / std

def denormalize_data(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Denormalize data using mean and std"""
    return data * std + mean

# HRRR normalization statistics (approximate values for native resolution)
HRRR_STATS = {
    "2t": {"mean": 285.0, "std": 15.0},      # 2m temperature (K)
    "10u": {"mean": 0.0, "std": 8.0},        # 10m u-wind (m/s)
    "10v": {"mean": 0.0, "std": 8.0},        # 10m v-wind (m/s)
    "msl": {"mean": 101325.0, "std": 1500.0}, # Mean sea level pressure (Pa)
    "cape": {"mean": 500.0, "std": 1000.0},   # CAPE (J/kg)
    "cin": {"mean": -50.0, "std": 100.0},     # CIN (J/kg)
    "refc": {"mean": 10.0, "std": 20.0},      # Radar reflectivity (dBZ)
    "hlcy": {"mean": 0.0, "std": 200.0},      # Helicity (m²/s²)
    "mxuphl": {"mean": 10.0, "std": 50.0},    # Max updraft helicity (m²/s²)
    
    # Atmospheric variables
    "t": {"mean": 250.0, "std": 30.0},        # Temperature (K)
    "u": {"mean": 0.0, "std": 15.0},          # U-wind (m/s)
    "v": {"mean": 0.0, "std": 15.0},          # V-wind (m/s)
    "q": {"mean": 0.005, "std": 0.008},       # Specific humidity (kg/kg)
    "z": {"mean": 5000.0, "std": 4000.0},     # Geopotential height (m)
    
    # Static variables
    "lsm": {"mean": 0.5, "std": 0.5},         # Land-sea mask
    "slt": {"mean": 0.1, "std": 0.3},         # Soil type
}

def get_normalization_stats(var_name: str) -> Tuple[float, float]:
    """Get normalization statistics for a variable"""
    if var_name in HRRR_STATS:
        stats = HRRR_STATS[var_name]
        return stats["mean"], stats["std"]
    else:
        logging.getLogger(__name__).warning(f"No normalization stats for {var_name}, using 0, 1")
        return 0.0, 1.0

def ensure_directories(*dirs: str):
    """Ensure directories exist"""
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)

def format_time(seconds: float) -> str:
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def count_parameters(model: torch.nn.Module) -> str:
    """Count and format model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if total >= 1e9:
        return f"{total/1e9:.1f}B total ({trainable/1e9:.1f}B trainable)"
    elif total >= 1e6:
        return f"{total/1e6:.1f}M total ({trainable/1e6:.1f}M trainable)"
    else:
        return f"{total:,} total ({trainable:,} trainable)"

if __name__ == "__main__":
    # Test utilities
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Test monitors
    gpu_monitor = GPUMonitor()
    system_monitor = SystemMonitor()
    
    logger.info("Testing utilities...")
    gpu_monitor.log_memory_usage()
    system_monitor.log_system_usage()
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(1)
    
    logger.info("Utilities test complete")