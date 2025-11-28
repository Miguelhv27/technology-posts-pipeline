#!/usr/bin/env python3
"""
Test runner script for the Technology Posts Pipeline
"""

import pytest
import sys
import os

def main():
    """Run all tests"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    exit_code = pytest.main([
        "-v",           
        "--tb=short",   
        "-x",          
        "tests/"        
    ])
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()