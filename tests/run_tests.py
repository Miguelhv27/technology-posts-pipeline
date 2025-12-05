#!/usr/bin/env python3
"""
Test Runner Script.
Ejecuta la suite completa de pruebas para el Pipeline DataOps.
"""

import pytest
import sys
import os

def main():
    """Configura el entorno y ejecuta pytest"""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f" Configurando PYTHONPATH: {project_root}")

    print(" Iniciando Suite de Pruebas DataOps...")

    exit_code = pytest.main([
        "-v",           
        "--tb=short",   
        "-W", "ignore::DeprecationWarning", 
        "tests/"        
    ])
    
    if exit_code == 0:
        print("\n TODOS LOS TESTS PASARON EXITOSAMENTE.")
    else:
        print(f"\n SE ENCONTRARON ERRORES (Exit Code: {exit_code})")
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()