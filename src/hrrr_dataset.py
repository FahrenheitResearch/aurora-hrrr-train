#!/usr/bin/env python3
"""
Native resolution HRRR dataset for Aurora training on H100
Handles full 1059×1799 HRRR grid with all convective variables
"""

import os
import torch
import xarray as xr
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass

from config import get_config
from utils import Timer, get_normalization_stats, validate_grid_dimensions

# Try to import Aurora classes, fallback to local definitions
try:
    from aurora import Batch, Metadata
    print("✅ Using Aurora Batch and Metadata classes")
except ImportError:
    print("⚠️  Using local Batch and Metadata classes")
    
    @dataclass
    class Metadata:
        """Aurora metadata structure"""
        lat: torch.Tensor
        lon: torch.Tensor
        time: torch.Tensor
        atmos_levels: torch.Tensor

    @dataclass
    class Batch:
        """Aurora batch structure"""
        surf_vars: Dict[str, torch.Tensor]
        atmos_vars: Dict[str, torch.Tensor]
        static_vars: Dict[str, torch.Tensor]
        metadata: Metadata

class HRRRNativeDataset(Dataset):
    """
    HRRR dataset for native resolution Aurora training
    
    Loads full resolution HRRR data (1059×1799) with all convective variables:
    - CAPE, CIN, Radar Reflectivity, Helicity, Max Updraft Helicity
    - Full atmospheric profile on pressure levels
    - Static variables (land-sea mask, orography, soil type)
    """
    
    def __init__(
        self, 
        data_dir: str,
        normalize: bool = True,
        cache_static: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.cache_static = cache_static
        self.logger = logging.getLogger(__name__)
        
        # Get configuration
        self.config = get_config()
        self.data_config = self.config["data"]
        self.model_config = self.config["model"]
        
        # Validate grid dimensions
        self.grid_height, self.grid_width = validate_grid_dimensions(
            self.data_config.crop_height,
            self.data_config.crop_width,
            self.model_config.patch_size
        )
        
        # Find HRRR file pairs
        self.file_pairs = self._find_hrrr_files()
        
        # Cache for static variables
        self._static_cache = None
        
        self.logger.info(f"HRRR Native Dataset initialized:")
        self.logger.info(f"  Data directory: {self.data_dir}")
        self.logger.info(f"  File pairs: {len(self.file_pairs)}")
        self.logger.info(f"  Grid size: {self.grid_height}×{self.grid_width}")
        self.logger.info(f"  Total pixels: {self.grid_height * self.grid_width:,}")
        self.logger.info(f"  Surface variables: {len(self.model_config.surf_vars)}")
        self.logger.info(f"  Atmospheric variables: {len(self.model_config.atmos_vars)}")
    
    def _find_hrrr_files(self) -> List[Tuple[Path, Path]]:
        """Find HRRR surface and pressure file pairs (GRIB2 or NetCDF)"""
        file_pairs = []
        
        # Try GRIB2 files first
        sfc_files = sorted(self.data_dir.glob("*_sfc.grib2"))
        for sfc_file in sfc_files:
            prs_file = sfc_file.with_name(sfc_file.name.replace("_sfc.grib2", "_prs.grib2"))
            if prs_file.exists():
                file_pairs.append((sfc_file, prs_file))
            else:
                self.logger.warning(f"No pressure file for {sfc_file.name}")
        
        # If no GRIB2 files, try NetCDF files (for testing)
        if not file_pairs:
            sfc_files = sorted(self.data_dir.glob("*_sfc.nc"))
            for sfc_file in sfc_files:
                prs_file = sfc_file.with_name(sfc_file.name.replace("_sfc.nc", "_prs.nc"))
                if prs_file.exists():
                    file_pairs.append((sfc_file, prs_file))
                    self.logger.info(f"Using NetCDF test files: {sfc_file.name}")
                else:
                    self.logger.warning(f"No pressure file for {sfc_file.name}")
        
        if not file_pairs:
            raise ValueError(f"No HRRR file pairs found in {self.data_dir} (tried .grib2 and .nc)")
        
        return file_pairs
    
    def _load_surface_variable(self, sfc_file: Path, var_name: str) -> Optional[np.ndarray]:
        """Load a single surface variable from HRRR file (GRIB2 or NetCDF)"""
        
        # Handle NetCDF test files differently
        if sfc_file.suffix == '.nc':
            return self._load_surface_variable_netcdf(sfc_file, var_name)
        
        var_config = self.data_config.hrrr_surf_vars.get(var_name)
        if not var_config:
            self.logger.warning(f"No configuration for surface variable: {var_name}")
            return None
        
        try:
            # Build filter for cfgrib
            filter_keys = {
                'typeOfLevel': var_config['level_type']
            }
            
            if 'level' in var_config:
                filter_keys['level'] = var_config['level']
            if 'top_level' in var_config:
                filter_keys['topLevel'] = var_config['top_level']
            if 'step_type' in var_config:
                filter_keys['stepType'] = var_config['step_type']
            if 'param' in var_config and var_config['param'] != 'unknown':
                filter_keys['shortName'] = var_config['param']
            
            # Load data
            ds = xr.open_dataset(sfc_file, engine='cfgrib', filter_by_keys=filter_keys)
            
            # Get variable data
            if var_config['param'] in ds.data_vars:
                data = ds[var_config['param']].values
            elif len(ds.data_vars) == 1:
                # If only one variable, use it (for unknown variables)
                var_key = list(ds.data_vars.keys())[0]
                data = ds[var_key].values
            else:
                self.logger.warning(f"Variable {var_name} not found in {sfc_file}")
                return None
            
            self.logger.debug(f"✅ Loaded {var_name}: {data.min():.1f} to {data.max():.1f}")
            return data
            
        except Exception as e:
            self.logger.warning(f"Failed to load {var_name} from {sfc_file}: {e}")
            return None
    
    def _load_surface_variable_netcdf(self, sfc_file: Path, var_name: str) -> Optional[np.ndarray]:
        """Load surface variable from NetCDF test file"""
        
        # Variable name mapping for test files
        var_mapping = {
            "2t": "2t", "10u": "10u", "10v": "10v", "msl": "mslma",
            "cape": "cape", "cin": "cin", "refc": "refc", "hlcy": "hlcy", 
            "mxuphl": "unknown"  # Max updraft helicity stored as 'unknown'
        }
        
        try:
            ds = xr.open_dataset(sfc_file)
            nc_var_name = var_mapping.get(var_name, var_name)
            
            if nc_var_name in ds.data_vars:
                data = ds[nc_var_name].values
                if data.ndim > 2:
                    data = data[0]  # Take first time step if present
                self.logger.debug(f"✅ Loaded {var_name} from NetCDF: {data.min():.1f} to {data.max():.1f}")
                return data
            else:
                self.logger.warning(f"Variable {var_name} ({nc_var_name}) not found in {sfc_file}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to load {var_name} from NetCDF {sfc_file}: {e}")
            return None
    
    def _load_atmospheric_variables(self, prs_file: Path) -> Dict[str, np.ndarray]:
        """Load atmospheric variables from pressure levels"""
        
        atmos_vars = {}
        
        for var_name in self.model_config.atmos_vars:
            var_config = self.data_config.hrrr_atmos_vars.get(var_name)
            if not var_config:
                continue
            
            try:
                ds = xr.open_dataset(
                    prs_file, 
                    engine='cfgrib',
                    filter_by_keys={
                        'typeOfLevel': 'isobaricInhPa',
                        'shortName': var_config['param']
                    }
                )
                
                if var_config['param'] in ds.data_vars:
                    full_data = ds[var_config['param']].values
                    
                    # Select pressure levels
                    pressure_levels = ds.isobaricInhPa.values
                    level_indices = []
                    
                    for target_level in self.data_config.pressure_levels:
                        # Find closest pressure level
                        closest_idx = np.argmin(np.abs(pressure_levels - target_level))
                        level_indices.append(closest_idx)
                    
                    # Extract selected levels
                    selected_data = full_data[level_indices]
                    atmos_vars[var_name] = selected_data
                    
                    self.logger.debug(f"✅ Loaded {var_name}: shape {selected_data.shape}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load atmospheric {var_name}: {e}")
        
        return atmos_vars
    
    def _get_static_variables(self, sfc_file: Path) -> Dict[str, torch.Tensor]:
        """Get static variables (cached)"""
        
        if self.cache_static and self._static_cache is not None:
            return self._static_cache
        
        static_vars = {}
        
        try:
            # Land-sea mask
            ds_lsm = xr.open_dataset(
                sfc_file, 
                engine='cfgrib',
                filter_by_keys={'typeOfLevel': 'surface', 'shortName': 'lsm'}
            )
            if 'lsm' in ds_lsm.data_vars:
                lsm_data = ds_lsm.lsm.values[:self.grid_height, :self.grid_width]
                static_vars['lsm'] = torch.tensor(lsm_data, dtype=torch.float32)
            
            # Orography (geopotential)
            ds_orog = xr.open_dataset(
                sfc_file,
                engine='cfgrib', 
                filter_by_keys={'typeOfLevel': 'surface', 'shortName': 'orog'}
            )
            if 'orog' in ds_orog.data_vars:
                orog_data = ds_orog.orog.values[:self.grid_height, :self.grid_width]
                static_vars['z'] = torch.tensor(orog_data, dtype=torch.float32)
            else:
                # Fallback to zeros
                static_vars['z'] = torch.zeros(self.grid_height, self.grid_width, dtype=torch.float32)
            
            # Soil type (if available)
            try:
                ds_soil = xr.open_dataset(
                    sfc_file,
                    engine='cfgrib',
                    filter_by_keys={'typeOfLevel': 'surface', 'shortName': 'slt'}
                )
                if 'slt' in ds_soil.data_vars:
                    soil_data = ds_soil.slt.values[:self.grid_height, :self.grid_width]
                    static_vars['slt'] = torch.tensor(soil_data, dtype=torch.float32)
                else:
                    static_vars['slt'] = torch.ones(self.grid_height, self.grid_width, dtype=torch.float32)
            except:
                static_vars['slt'] = torch.ones(self.grid_height, self.grid_width, dtype=torch.float32)
            
        except Exception as e:
            self.logger.warning(f"Failed to load static variables: {e}")
            # Create fallback static variables
            static_vars = {
                'lsm': torch.ones(self.grid_height, self.grid_width, dtype=torch.float32),
                'z': torch.zeros(self.grid_height, self.grid_width, dtype=torch.float32),
                'slt': torch.ones(self.grid_height, self.grid_width, dtype=torch.float32)
            }
        
        if self.cache_static:
            self._static_cache = static_vars
        
        return static_vars
    
    def _get_coordinates(self, sfc_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Get latitude and longitude coordinates"""
        
        try:
            # Load any surface variable to get coordinates
            ds = xr.open_dataset(
                sfc_file,
                engine='cfgrib',
                filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2}
            )
            
            lat = ds.latitude.values[:self.grid_height, :self.grid_width]
            lon = ds.longitude.values[:self.grid_height, :self.grid_width]
            
            # Ensure longitude is in 0-360 range
            lon = np.where(lon < 0, lon + 360, lon)
            
            return lat, lon
            
        except Exception as e:
            self.logger.error(f"Failed to get coordinates: {e}")
            # Fallback coordinates
            lat = np.linspace(25, 53, self.grid_height)
            lon = np.linspace(235, 295, self.grid_width)
            lat, lon = np.meshgrid(lat, lon, indexing='ij')
            return lat, lon
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Batch:
        """Load a single HRRR sample"""
        
        sfc_file, prs_file = self.file_pairs[idx]
        
        with Timer(f"Load sample {idx}"):
            # Load surface variables
            surf_vars = {}
            for var_name in self.model_config.surf_vars:
                data = self._load_surface_variable(sfc_file, var_name)
                
                if data is not None:
                    # Crop to target size
                    cropped_data = data[:self.grid_height, :self.grid_width]
                    
                    # Normalize if requested
                    if self.normalize:
                        mean, std = get_normalization_stats(var_name)
                        cropped_data = (cropped_data - mean) / std
                    
                    surf_vars[var_name] = torch.tensor(cropped_data, dtype=torch.float32)
                else:
                    # Fallback to zeros
                    self.logger.warning(f"Using zeros for missing variable: {var_name}")
                    surf_vars[var_name] = torch.zeros(self.grid_height, self.grid_width, dtype=torch.float32)
            
            # Load atmospheric variables
            atmos_data = self._load_atmospheric_variables(prs_file)
            atmos_vars = {}
            
            for var_name in self.model_config.atmos_vars:
                if var_name in atmos_data:
                    data = atmos_data[var_name]
                    # Crop spatial dimensions
                    cropped_data = data[:, :self.grid_height, :self.grid_width]
                    
                    # Normalize if requested
                    if self.normalize:
                        mean, std = get_normalization_stats(var_name)
                        cropped_data = (cropped_data - mean) / std
                    
                    # Aurora expects 3D tensors [levels, height, width]
                    atmos_vars[var_name] = torch.tensor(cropped_data, dtype=torch.float32)
                else:
                    # Fallback to zeros
                    num_levels = len(self.data_config.pressure_levels)
                    atmos_vars[var_name] = torch.zeros(num_levels, self.grid_height, self.grid_width, dtype=torch.float32)
            
            # Get static variables
            static_vars = self._get_static_variables(sfc_file)
            
            # Get coordinates
            lat, lon = self._get_coordinates(sfc_file)
            
            # Create metadata
            metadata = Metadata(
                lat=torch.tensor(lat, dtype=torch.float32),
                lon=torch.tensor(lon, dtype=torch.float32),
                time=torch.tensor([0.0], dtype=torch.float32),  # Relative time
                atmos_levels=torch.tensor(self.data_config.pressure_levels, dtype=torch.float32)
            )
            
            # Create batch
            batch = Batch(
                surf_vars=surf_vars,
                atmos_vars=atmos_vars,
                static_vars=static_vars,
                metadata=metadata
            )
        
        return batch

def create_dataloader(
    dataset: HRRRNativeDataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> torch.utils.data.DataLoader:
    """Create DataLoader for HRRR dataset"""
    
    def collate_fn(batch_list):
        """Custom collate function for Aurora batches"""
        if len(batch_list) == 1:
            return batch_list[0]
        
        # For batch_size > 1, we need to stack tensors
        # This is complex for Aurora format, so for now we only support batch_size=1
        raise NotImplementedError("Batch size > 1 not implemented for native resolution")
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )

if __name__ == "__main__":
    # Test dataset
    import sys
    from utils import setup_logging
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "hrrr_data"
    
    logger.info(f"Testing HRRR Native Dataset with data from: {data_dir}")
    
    try:
        dataset = HRRRNativeDataset(data_dir)
        logger.info(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test loading one sample
        if len(dataset) > 0:
            with Timer("Load first sample"):
                sample = dataset[0]
            
            logger.info("Sample loaded successfully:")
            logger.info(f"  Surface variables: {list(sample.surf_vars.keys())}")
            logger.info(f"  Atmospheric variables: {list(sample.atmos_vars.keys())}")
            logger.info(f"  Static variables: {list(sample.static_vars.keys())}")
            logger.info(f"  Grid shape: {sample.surf_vars['2t'].shape}")
            
            # Test dataloader
            dataloader = create_dataloader(dataset, batch_size=1, num_workers=0)
            logger.info("DataLoader created successfully")
    
    except Exception as e:
        logger.error(f"Dataset test failed: {e}")
        sys.exit(1)