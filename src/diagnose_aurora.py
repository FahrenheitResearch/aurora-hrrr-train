#!/usr/bin/env python3
"""
Diagnose Aurora import issues
"""

import sys
import os

def diagnose_aurora():
    """Diagnose Aurora package installation"""
    
    print("🔍 Aurora Package Diagnosis")
    print("=" * 40)
    
    # Check if aurora package exists
    try:
        import aurora
        print(f"✅ Aurora package found at: {aurora.__file__}")
        print(f"   Aurora version: {getattr(aurora, '__version__', 'Unknown')}")
    except ImportError as e:
        print(f"❌ Aurora package not found: {e}")
        return False
    
    # Check package contents
    print(f"\n📦 Aurora package contents:")
    for item in sorted(dir(aurora)):
        if not item.startswith('_'):
            try:
                attr = getattr(aurora, item)
                attr_type = type(attr).__name__
                print(f"   {item:<20} ({attr_type})")
            except:
                print(f"   {item:<20} (unable to inspect)")
    
    # Check specific classes
    print(f"\n🔍 Checking specific classes:")
    
    classes_to_check = [
        'Aurora',
        'AuroraPretrained', 
        'AuroraSmall',
        'Batch',
        'Metadata'
    ]
    
    for class_name in classes_to_check:
        try:
            cls = getattr(aurora, class_name)
            print(f"   ✅ {class_name}: {cls}")
        except AttributeError:
            print(f"   ❌ {class_name}: Not found")
    
    # Check submodules
    print(f"\n📁 Checking submodules:")
    
    submodules = ['normalisation', 'model', 'batch']
    
    for module_name in submodules:
        try:
            module = getattr(aurora, module_name)
            print(f"   ✅ {module_name}: {module}")
        except AttributeError:
            print(f"   ❌ {module_name}: Not found")
    
    # Try specific imports
    print(f"\n🧪 Testing specific imports:")
    
    imports_to_test = [
        ("from aurora import AuroraPretrained", "AuroraPretrained"),
        ("from aurora import Batch", "Batch"),  
        ("from aurora import Metadata", "Metadata"),
        ("from aurora.normalisation import locations", "normalisation.locations"),
        ("from aurora.normalisation import scales", "normalisation.scales")
    ]
    
    for import_stmt, description in imports_to_test:
        try:
            exec(import_stmt)
            print(f"   ✅ {description}")
        except Exception as e:
            print(f"   ❌ {description}: {e}")
    
    # Check installation method
    print(f"\n💻 Installation info:")
    try:
        import pkg_resources
        aurora_dist = pkg_resources.get_distribution('microsoft-aurora')
        print(f"   Package: {aurora_dist.project_name}")
        print(f"   Version: {aurora_dist.version}")
        print(f"   Location: {aurora_dist.location}")
    except:
        print("   Could not get package info")
    
    return True

def suggest_fixes():
    """Suggest potential fixes"""
    
    print(f"\n🛠️  Potential fixes:")
    print("1. Reinstall Aurora:")
    print("   pip uninstall microsoft-aurora")
    print("   pip install microsoft-aurora")
    print()
    print("2. Check Python environment:")
    print("   which python")
    print("   which pip")
    print("   conda list | grep aurora")
    print()
    print("3. Try alternative installation:")
    print("   conda install -c conda-forge microsoft-aurora")
    print()
    print("4. Check for conflicts:")
    print("   pip list | grep aurora")
    print("   pip check")

if __name__ == "__main__":
    try:
        if diagnose_aurora():
            print("\n✅ Aurora package appears to be properly installed")
        else:
            print("\n❌ Aurora package has issues")
            suggest_fixes()
    except Exception as e:
        print(f"\n💥 Diagnosis failed: {e}")
        suggest_fixes()