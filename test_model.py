
#!/usr/bin/env python3
"""
Simple test script to verify model training and chat functionality
"""

import os
import sys

def test_training_data():
    """Test training data format"""
    print("🧪 Testing training data format...")
    
    if not os.path.exists('user_training_data.jsonl'):
        print("❌ user_training_data.jsonl not found!")
        return False
    
    try:
        import json
        
        with open('user_training_data.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        if len(lines) == 0:
            print("❌ Training data is empty!")
            return False
            
        # Test first few lines
        for i, line in enumerate(lines[:3]):
            data = json.loads(line.strip())
            
            required_fields = ['input', 'output', 'tone', 'style_id']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Line {i+1} missing field: {field}")
                    return False
        
        print(f"✅ Training data format OK! ({len(lines)} samples)")
        return True
        
    except Exception as e:
        print(f"❌ Training data test failed: {e}")
        return False

def test_imports():
    """Test if all imports work"""
    print("🧪 Testing imports...")
    
    try:
        sys.path.append('./model')
        
        import torch
        print("  ✅ PyTorch imported")
        
        from model.custom_tokenizer import UltraChronoFireTokenizer
        print("  ✅ Custom tokenizer imported")
        
        from model.components import ToneDetector
        print("  ✅ Components imported")
        
        from model.model import UltraChronoFireTransformer
        print("  ✅ Model imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("🧪 Testing model creation...")
    
    try:
        sys.path.append('./model')
        import torch
        from model.model import UltraChronoFireTransformer
        
        model = UltraChronoFireTransformer(
            vocab_size=1000,  # Small vocab for testing
            d_model=256,      # Small model for testing
            num_layers=2,
            k_window=8,
            num_heads=4,
            personality_mode='fixed'
        )
        
        print("✅ Model created successfully!")
        print(f"📊 Model stats: {model.get_performance_stats()}")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔥 Ultra ChronoFire Model Test Suite")
    print("=" * 50)
    
    tests = [
        ("Training Data Format", test_training_data),
        ("Import Test", test_imports),
        ("Model Creation", test_model_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running test: {test_name}")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready for training.")
        print("\n🚀 Next steps:")
        print("1. Click the Run button to start training")
        print("2. After training, use: 'Chat Interface' workflow")
    else:
        print("❌ Some tests failed. Please fix issues before training.")
    
    return passed == total

if __name__ == "__main__":
    main()
