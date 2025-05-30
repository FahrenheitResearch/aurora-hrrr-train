#!/usr/bin/env python3
"""
Basic Aurora test with minimal data to check if the model works at all
"""

import torch
from aurora import Aurora, Batch, Metadata

def test_aurora_basic():
    print("Testing basic Aurora functionality...")
    
    # Create Aurora model with defaults
    model = Aurora()
    model.eval()
    
    print(f"Aurora model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create minimal test data
    height, width = 64, 64  # Small grid for testing
    
    # Surface variables (Aurora defaults)
    surf_vars = {
        "2t": torch.randn(height, width),
        "10u": torch.randn(height, width), 
        "10v": torch.randn(height, width),
        "msl": torch.randn(height, width)
    }
    
    # Atmospheric variables - try different naming formats
    pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    atmos_vars = {}
    
    # Test 1: Try without pressure level suffixes (3D tensors)
    for var in ["z", "u", "v", "t", "q"]:
        atmos_vars[var] = torch.randn(len(pressure_levels), height, width)
    
    # Static variables
    static_vars = {
        "lsm": torch.randn(height, width),
        "z": torch.randn(height, width),
        "slt": torch.randn(height, width)
    }
    
    # Metadata
    lat = torch.linspace(25, 53, height).unsqueeze(1).repeat(1, width)
    lon = torch.linspace(235, 295, width).unsqueeze(0).repeat(height, 1)
    
    metadata = Metadata(
        lat=lat,
        lon=lon,
        time=torch.tensor([0.0]),
        atmos_levels=torch.tensor(pressure_levels, dtype=torch.float32)
    )
    
    # Create batch
    batch = Batch(
        surf_vars=surf_vars,
        atmos_vars=atmos_vars,
        static_vars=static_vars,
        metadata=metadata
    )
    
    print(f"Created batch with:")
    print(f"  Surface vars: {list(surf_vars.keys())}")
    print(f"  Atmospheric vars: {list(atmos_vars.keys())}")
    print(f"  Static vars: {list(static_vars.keys())}")
    print(f"  Atmos shapes: {[(k, v.shape) for k, v in atmos_vars.items()]}")
    
    try:
        with torch.no_grad():
            output = model(batch)
        print("✅ Aurora forward pass successful!")
        print(f"Output type: {type(output)}")
        return True
        
    except Exception as e:
        print(f"❌ Aurora forward pass failed: {e}")
        return False

if __name__ == "__main__":
    test_aurora_basic()