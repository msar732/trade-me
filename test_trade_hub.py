#!/usr/bin/env python3
"""
Test script for Trade Hub application
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test main application
        from trade_hub import app, TradeHubConfig, User, Listing, Category
        print("✅ Main application imports successful")
        
        # Test routes (import specific functions)
        import trade_hub_routes
        print("✅ Routes imports successful")
        
        # Test templates
        from trade_hub_templates import TRADE_HUB_HTML_TEMPLATE, LOGIN_HTML_TEMPLATE, REGISTER_HTML_TEMPLATE
        print("✅ Templates imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration"""
    try:
        print("\nTesting configuration...")
        
        from trade_hub import TradeHubConfig
        
        # Test config values
        assert hasattr(TradeHubConfig, 'DATABASE_URL')
        assert hasattr(TradeHubConfig, 'SECRET_KEY')
        assert hasattr(TradeHubConfig, 'AI_ENABLED')
        assert hasattr(TradeHubConfig, 'REAL_TIME_CHAT')
        
        print("✅ Configuration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_database_models():
    """Test database models"""
    try:
        print("\nTesting database models...")
        
        from trade_hub import User, Listing, Category, Message, Review, Watchlist, Notification
        
        # Test model attributes
        assert hasattr(User, 'username')
        assert hasattr(User, 'email')
        assert hasattr(Listing, 'title')
        assert hasattr(Listing, 'price')
        assert hasattr(Category, 'name')
        
        print("✅ Database models test passed")
        return True
        
    except Exception as e:
        print(f"❌ Database models error: {e}")
        return False

def test_ai_services():
    """Test AI services"""
    try:
        print("\nTesting AI services...")
        
        from trade_hub import AIRecommendationEngine, PricePredictionEngine
        
        # Test AI engines
        rec_engine = AIRecommendationEngine()
        price_engine = PricePredictionEngine()
        
        assert hasattr(rec_engine, 'train_model')
        assert hasattr(rec_engine, 'get_recommendations')
        assert hasattr(price_engine, 'train_model')
        assert hasattr(price_engine, 'predict_price')
        
        print("✅ AI services test passed")
        return True
        
    except Exception as e:
        print(f"❌ AI services error: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    try:
        print("\nTesting utility functions...")
        
        from trade_hub import validate_password_strength, calculate_ai_score, allowed_file
        
        # Test password validation
        is_strong, msg = validate_password_strength("Test123!")
        assert is_strong == True
        
        is_weak, msg = validate_password_strength("123")
        assert is_weak == False
        
        # Test file validation
        assert allowed_file("test.jpg") == True
        assert allowed_file("test.txt") == False
        
        print("✅ Utility functions test passed")
        return True
        
    except Exception as e:
        print(f"❌ Utility functions error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Trade Hub Application Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_database_models,
        test_ai_services,
        test_utility_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Trade Hub is ready to run.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)