#!/usr/bin/env python3
"""
Create synthetic HRRR test data for Aurora training
"""

import numpy as np
import xarray as xr
import torch
from pathlib import Path
import argparse
from datetime import datetime, timedelta

def create_synthetic_hrrr_file(output_path, file_type="surface", grid_size=(1056, 1796)):
    """Create synthetic HRRR GRIB-like data as NetCDF for testing"""
    
    height, width = grid_size
    
    # Create coordinate arrays
    lat = np.linspace(20, 60, height)
    lon = np.linspace(230, 300, width)
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing='ij')
    
    # Create time coordinate
    time = np.array([datetime.now()], dtype='datetime64[ns]')
    
    if file_type == "surface":
        # Surface variables
        data_vars = {
            # Basic meteorological
            '2t': (['y', 'x'], 280 + 20 * np.random.randn(height, width)),  # 2m temperature
            '10u': (['y', 'x'], 5 * np.random.randn(height, width)),        # 10m u-wind
            '10v': (['y', 'x'], 3 * np.random.randn(height, width)),        # 10m v-wind
            'mslma': (['y', 'x'], 101325 + 1000 * np.random.randn(height, width)),  # MSL pressure
            
            # Convective variables
            'cape': (['y', 'x'], np.abs(np.random.exponential(500, (height, width)))),  # CAPE
            'cin': (['y', 'x'], -np.abs(np.random.exponential(50, (height, width)))),   # CIN (negative)
            'refc': (['y', 'x'], -10 + 30 * np.random.rand(height, width)),            # Reflectivity
            'hlcy': (['y', 'x'], 200 * np.random.randn(height, width)),                # Helicity
            
            # Static variables
            'lsm': (['y', 'x'], (np.random.rand(height, width) > 0.3).astype(float)), # Land-sea mask
            'orog': (['y', 'x'], np.abs(1000 * np.random.randn(height, width))),      # Orography
            'slt': (['y', 'x'], np.ones((height, width))),                            # Soil type
        }
        
        # Add unknown variables for max updraft helicity
        data_vars['unknown'] = (['y', 'x'], 25 * np.random.exponential(1, (height, width)))
        
    else:  # pressure levels
        # Pressure levels (standard HRRR levels)
        pressure_levels = np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
        
        data_vars = {
            't': (['isobaricInhPa', 'y', 'x'], 
                  250 + 30 * np.random.randn(len(pressure_levels), height, width)),
            'u': (['isobaricInhPa', 'y', 'x'], 
                  15 * np.random.randn(len(pressure_levels), height, width)),
            'v': (['isobaricInhPa', 'y', 'x'], 
                  10 * np.random.randn(len(pressure_levels), height, width)),
            'q': (['isobaricInhPa', 'y', 'x'], 
                  0.01 * np.abs(np.random.randn(len(pressure_levels), height, width))),
            'gh': (['isobaricInhPa', 'y', 'x'], 
                   5000 + 4000 * np.random.randn(len(pressure_levels), height, width)),
        }
    
    # Create coordinates
    coords = {
        'latitude': (['y', 'x'], lat_2d),
        'longitude': (['y', 'x'], lon_2d),
        'time': time,
        'y': np.arange(height),
        'x': np.arange(width)
    }
    
    if file_type == "pressure":
        coords['isobaricInhPa'] = pressure_levels
    
    # Create dataset
    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    
    # Add attributes to make it look like HRRR
    ds.attrs = {
        'title': f'Synthetic HRRR {file_type} data for testing',
        'institution': 'Synthetic data for Aurora training',
        'source': 'create_test_data.py'
    }
    
    # Save as NetCDF (easier than GRIB for testing)
    ds.to_netcdf(output_path)
    print(f"✅ Created synthetic {file_type} file: {output_path}")

def create_test_dataset(output_dir="hrrr_data", num_timesteps=4, grid_size=(256, 256)):
    """Create a complete test dataset"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🧪 Creating synthetic HRRR test dataset")
    print(f"   Output: {output_dir}")
    print(f"   Timesteps: {num_timesteps}")
    print(f"   Grid size: {grid_size[0]}×{grid_size[1]}")
    
    base_date = datetime.now() - timedelta(days=1)
    
    for i in range(num_timesteps):
        current_date = base_date + timedelta(hours=i)
        date_str = current_date.strftime('%Y%m%d')
        hour_str = current_date.strftime('%H')
        
        # Surface file
        sfc_filename = f"hrrr_{date_str}_{hour_str}z_f00_sfc.nc"
        sfc_path = output_dir / sfc_filename
        create_synthetic_hrrr_file(sfc_path, "surface", grid_size)
        
        # Pressure file
        prs_filename = f"hrrr_{date_str}_{hour_str}z_f00_prs.nc"
        prs_path = output_dir / prs_filename
        create_synthetic_hrrr_file(prs_path, "pressure", grid_size)
    
    print(f"\n🎉 Test dataset created!")
    print(f"📁 Files in {output_dir}:")
    for file in sorted(output_dir.glob("*.nc")):
        print(f"   - {file.name}")
    
    print(f"\n🚀 Test the dataset:")
    print(f"   python hrrr_dataset.py {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create synthetic HRRR test data")
    parser.add_argument("--output-dir", default="hrrr_test_data", help="Output directory")
    parser.add_argument("--num-timesteps", type=int, default=4, help="Number of timesteps")
    parser.add_argument("--grid-height", type=int, default=256, help="Grid height (should be divisible by 4)")
    parser.add_argument("--grid-width", type=int, default=256, help="Grid width (should be divisible by 4)")
    
    args = parser.parse_args()
    
    # Ensure grid dimensions are divisible by 4 (Aurora requirement)
    height = (args.grid_height // 4) * 4
    width = (args.grid_width // 4) * 4
    
    if height != args.grid_height or width != args.grid_width:
        print(f"⚠️  Adjusted grid size to {height}×{width} (must be divisible by 4)")
    
    create_test_dataset(
        output_dir=args.output_dir,
        num_timesteps=args.num_timesteps,
        grid_size=(height, width)
    )

if __name__ == "__main__":
    main()