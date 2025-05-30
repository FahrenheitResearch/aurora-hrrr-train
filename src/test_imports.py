#!/usr/bin/env python3
"""
Test script to verify all imports work correctly on H100 system
"""

import sys
import traceback

def test_basic_imports():
    """Test basic Python imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        import xarray as xr
        import cfgrib
        print("✅ Scientific computing packages")
    except Exception as e:
        print(f"❌ Scientific packages failed: {e}")
        return False
    
    return True

def test_aurora_imports():
    """Test Aurora imports"""
    print("\n🌌 Testing Aurora imports...")
    
    try:
        import aurora
        aurora_classes = [x for x in dir(aurora) if 'Aurora' in x]
        print(f"✅ Aurora package imported")
        print(f"   Available classes: {aurora_classes}")
        
        # Try to import any available Aurora model class
        model_class = None
        for class_name in ['Aurora', 'AuroraHighRes', 'AuroraSmall', 'AuroraPretrained']:
            try:
                model_class = getattr(aurora, class_name)
                print(f"✅ {class_name} imported successfully")
                break
            except AttributeError:
                continue
        
        if model_class is None:
            print("❌ No Aurora model class found")
            return False
        
        from aurora import Batch, Metadata
        print("✅ Batch and Metadata imported")
        
        from aurora.normalisation import locations, scales
        print("✅ Normalization modules imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Aurora import failed: {e}")
        print("This is expected if Aurora is not properly installed")
        return False
    except Exception as e:
        print(f"❌ Aurora import error: {e}")
        traceback.print_exc()
        return False

def test_local_imports():
    """Test local module imports"""
    print("\n📦 Testing local module imports...")
    
    try:
        from config import get_config
        print("✅ Config module imported")
        
        config = get_config()
        print(f"   Model config: {len(config['model'].surf_vars)} surface variables")
        
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from utils import setup_logging, GPUMonitor
        print("✅ Utils module imported")
        
        setup_logging()
        print("✅ Logging setup successful")
        
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from hrrr_dataset import HRRRNativeDataset
        print("✅ HRRR dataset module imported")
        
    except Exception as e:
        print(f"❌ HRRR dataset import failed: {e}")
        traceback.print_exc()
        return False
    
    try:
        from aurora_native import AuroraNative
        print("✅ Aurora native module imported")
        
    except Exception as e:
        print(f"❌ Aurora native import failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\n🏗️  Testing model creation...")
    
    try:
        import torch
        from aurora_native import create_aurora_native
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create model without pretrained weights for testing
        model = create_aurora_native(pretrained=False, device=device)
        print("✅ Aurora native model created successfully")
        
        # Test memory usage calculation
        memory_info = model.get_memory_usage()
        print("✅ Memory usage calculation successful")
        print(f"   Estimated memory: {memory_info.get('total_estimated_gb', 'N/A')} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Aurora H100 Import Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Aurora Imports", test_aurora_imports),
        ("Local Module Imports", test_local_imports),
        ("Model Creation", test_model_creation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 All tests passed! The environment is ready for Aurora training.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        print("\n💡 Common solutions:")
        print("1. Make sure you activated the conda environment: conda activate aurora_h100")
        print("2. If Aurora import fails, try: pip install --upgrade microsoft-aurora")
        print("3. Check CUDA installation: nvidia-smi")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())