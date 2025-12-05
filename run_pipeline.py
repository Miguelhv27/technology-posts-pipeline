#!/usr/bin/env python3
"""
Main Execution Entry Point for Technology Posts Pipeline.
Version: 2.0 (Refactored for DataOps)
"""

import os
import sys
import logging
import argparse

# Asegurar que el directorio src sea visible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator import Orchestrator, configure_logging

def parse_arguments():
    """
    Argumentos de línea de comandos simples.
    La lógica compleja de pasos se maneja ahora en el config.yaml.
    """
    parser = argparse.ArgumentParser(description='Technology Posts DataOps Pipeline')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/pipeline_config.yaml',
        help='Ruta al archivo de configuración YAML'
    )
    
    return parser.parse_args()

def check_environment():
    """
    Verificaciones básicas antes de arrancar.
    (Las verificaciones profundas las hace validation.py)
    """
    # 1. Verificar carpetas críticas
    required_dirs = ['data', 'config', 'src']
    for d in required_dirs:
        if not os.path.exists(d):
            print(f"ERROR: Directorio crítico no encontrado: {d}/")
            return False
            
    # 2. Verificar archivo de configuración
    if not os.path.exists("config/pipeline_config.yaml"):
        print("ERROR: config/pipeline_config.yaml no encontrado.")
        return False

    return True

def main():
    """Main execution flow"""
    # 1. Parsear argumentos
    args = parse_arguments()
    
    # 2. Configurar Logging Centralizado
    # (Usamos la función del orquestador para consistencia)
    configure_logging()
    
    logging.info("========================================")
    logging.info("   TECHNOLOGY POSTS DATA PIPELINE 2.0   ")
    logging.info("========================================")
    
    # 3. Chequeo de Entorno
    if not check_environment():
        logging.critical("El entorno no está listo. Abortando.")
        sys.exit(1)

    try:
        # 4. Inicializar Orquestador
        logging.info(f"Cargando configuración desde: {args.config}")
        orchestrator = Orchestrator(config_path=args.config)
        
        # 5. Ejecutar Pipeline
        # El orquestador maneja la secuencia Ingesta -> Validación -> NLP -> Análisis
        success = orchestrator.run()
        
        if success:
            logging.info(" Pipeline finalizado EXITOSAMENTE.")
            sys.exit(0)
        else:
            logging.error(" El pipeline finalizó con ERRORES (ver logs).")
            sys.exit(1)
            
    except Exception as e:
        logging.critical(f" Error fatal no controlado en main: {e}")
        # Imprimir traceback para debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()