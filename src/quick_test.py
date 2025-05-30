#!/usr/bin/env python3
"""
Quick test to verify Aurora setup on H100
"""

def main():
    print("🧪 Quick Aurora H100 Test")
    print("=" * 40)
    
    # Test basic imports
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
    except:
        print("❌ PyTorch import failed")
        return False
    
    # Test Aurora
    try:
        import aurora
        aurora_classes = [x for x in dir(aurora) if 'Aurora' in x]
        print(f"✅ Aurora package")
        print(f"   Classes: {aurora_classes}")
        
        # Test what we can actually use
        working_class = None
        for class_name in ['Aurora', 'AuroraHighRes', 'AuroraSmall']:
            try:
                cls = getattr(aurora, class_name)
                working_class = class_name
                print(f"✅ Can use: {class_name}")
                break
            except:
                pass
        
        if working_class:
            print(f"🎯 Recommended: Use {working_class} instead of AuroraPretrained")
        else:
            print("❌ No working Aurora class found")
            return False
            
    except ImportError as e:
        print(f"❌ Aurora import failed: {e}")
        return False
    
    # Test our code
    try:
        from aurora_native import AuroraNative
        print("✅ Our aurora_native module works")
    except Exception as e:
        print(f"❌ aurora_native failed: {e}")
        return False
    
    print("\n🎉 Everything looks good!")
    print("\n📋 Next steps:")
    print("1. Download HRRR data:")
    print("   python download_hrrr_data.py --start-date 2023-07-15 --end-date 2023-07-15")
    print("2. Test dataset:")
    print("   python hrrr_dataset.py hrrr_data")
    print("3. Train model:")
    print("   python train_aurora_native.py --data-dir hrrr_data --debug")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 If you see import errors, run:")
        print("   python test_imports.py")
        print("   python diagnose_aurora.py")
        exit(1)