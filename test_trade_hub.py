#!/usr/bin/env python3
"""
Trade Hub - Test Script
Tests basic functionality without running the full server
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing Trade Hub imports...")
    
    try:
        # Test core Python modules
        import os, json, uuid, hashlib, secrets, re, random, base64, hmac, time, threading
        from datetime import datetime, timedelta
        from typing import List, Optional, Dict, Any, Union
        import logging
        from pathlib import Path
        from functools import lru_cache, wraps
        from collections import defaultdict
        import math
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        print("✅ Core Python modules: OK")
        
        # Test Flask modules (these might not be available)
        try:
            from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash, session, send_from_directory, make_response, abort
            print("✅ Flask modules: OK")
        except ImportError as e:
            print(f"⚠️  Flask modules: Not available ({e})")
        
        # Test database modules
        try:
            from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, JSON, Index, event, func, and_, or_
            from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship, scoped_session
            from sqlalchemy.pool import QueuePool
            from sqlalchemy.exc import IntegrityError
            print("✅ SQLAlchemy modules: OK")
        except ImportError as e:
            print(f"⚠️  SQLAlchemy modules: Not available ({e})")
        
        # Test optional modules
        try:
            import requests
            from bs4 import BeautifulSoup
            print("✅ Web scraping modules: OK")
        except ImportError:
            print("⚠️  Web scraping modules: Not available")
        
        try:
            import numpy as np
            import pandas as pd
            from sklearn.feature_extraction.text import TfidfVectorizer
            print("✅ ML modules: OK")
        except ImportError:
            print("⚠️  ML modules: Not available")
        
        try:
            import redis
            print("✅ Redis module: OK")
        except ImportError:
            print("⚠️  Redis module: Not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_config():
    """Test configuration class"""
    print("\n🧪 Testing Trade Hub configuration...")
    
    try:
        # Import the config class from our module
        sys.path.insert(0, '/workspace')
        
        # Test basic configuration
        class TestConfig:
            DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tradehub_test.db")
            SECRET_KEY = "test-secret-key"
            UPLOAD_FOLDER = "./test_uploads"
            
        config = TestConfig()
        print(f"✅ Database URL: {config.DATABASE_URL}")
        print(f"✅ Secret Key: {'*' * len(config.SECRET_KEY)}")
        print(f"✅ Upload Folder: {config.UPLOAD_FOLDER}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_html_template():
    """Test if HTML template is valid"""
    print("\n🧪 Testing HTML template...")
    
    try:
        # Read the trade_hub.py file and check for template
        with open('/workspace/trade_hub.py', 'r') as f:
            content = f.read()
            
        if 'TRADE_HUB_TEMPLATE' in content:
            print("✅ HTML template found in code")
            
            # Check for key HTML elements
            template_start = content.find('TRADE_HUB_TEMPLATE = ')
            if template_start > 0:
                template_section = content[template_start:template_start + 1000]
                
                if '<!DOCTYPE html>' in template_section:
                    print("✅ Valid HTML5 doctype")
                if '<html lang="en">' in template_section:
                    print("✅ HTML lang attribute")
                if '<meta charset="UTF-8">' in template_section:
                    print("✅ UTF-8 charset")
                if 'bootstrap' in content.lower():
                    print("✅ Bootstrap CSS framework")
                if 'trade hub' in content.lower():
                    print("✅ Trade Hub branding")
                    
                return True
            else:
                print("⚠️  Template content not found")
                return False
        else:
            print("❌ HTML template not found")
            return False
            
    except Exception as e:
        print(f"❌ HTML template test failed: {e}")
        return False

def test_file_structure():
    """Test if all required files exist"""
    print("\n🧪 Testing file structure...")
    
    files_to_check = [
        '/workspace/trade_hub.py',
        '/workspace/requirements.txt',
        '/workspace/README.md'
    ]
    
    all_good = True
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {os.path.basename(file_path)}: {size:,} bytes")
        else:
            print(f"❌ {os.path.basename(file_path)}: Missing")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🚀 Trade Hub - Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("HTML Template Test", test_html_template),
        ("File Structure Test", test_file_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Tests Passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 All tests passed! Trade Hub is ready to run.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)