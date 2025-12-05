import os
import yaml
import logging
import json
import sys
from datetime import datetime

# Importacion de modulos del proyecto refactorizados
from src.data_ingestion import ingest_data
from src.data_validation import validate_raw_data
from src.data_transformation import transform_data
from src.nlp import run_nlp_pipeline
from src.analysis import run_statistical_analysis
from src.network_analysis import run_network_analysis

def configure_logging(log_dir="logs/"):
    """Configuracion centralizada de logs"""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Configuración del handler de archivo con encoding utf-8 explícito para Windows
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(module)s | %(message)s"))

    # Configuración del handler de consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # Configuración básica (reseteando handlers previos para evitar duplicados)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    
    logging.info(f"Logging iniciado en: {log_path}")

class Orchestrator:
    """
    DataOps Pipeline Orchestrator.
    Coordina el flujo de datos desde la ingesta hasta el analisis y reporte.
    """

    def __init__(self, config_path="config/pipeline_config.yaml"):
        # Carga la configuración al iniciar
        self.config = self.load_config(config_path)
        
        # Define rutas basadas en la configuración
        self.processed_data_dir = self.config["paths"]["processed_data_dir"]
        self.raw_data_dir = self.config["paths"]["raw_data_dir"]
        self.logs_dir = self.config["paths"]["logs_dir"]
        self.outputs_dir = self.config["paths"].get("outputs_dir", "outputs/")
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_config(self, path):
        """Carga el archivo YAML de configuración con encoding UTF-8"""
        if not os.path.exists(path):
            print(f"CRITICAL: Config file not found at {path}")
            sys.exit(1)
        
        # AQUÍ ESTABA EL ERROR: Agregamos encoding='utf-8'
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def create_output_folders(self):
        """Crea estructura de carpetas para esta ejecucion"""
        # Versionamiento por timestamp si esta habilitado
        if self.config["pipeline"].get("create_timestamped_folders", True):
            run_folder = os.path.join(self.outputs_dir, self.current_run_id)
        else:
            run_folder = self.outputs_dir
            
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(run_folder, exist_ok=True)
        
        logging.info(f"Directorios de salida preparados en: {run_folder}")
        return run_folder

    def run_ingestion(self):
        """Paso 1: Ingesta"""
        if not self.config["ingestion"]["enabled"]:
            logging.warning("Ingesta desactivada. Buscando datos existentes...")
            return {}

        logging.info(">>> PASO 1: INGESTA DE DATOS")
        # Retorna diccionario de rutas {'twitter': path, 'reddit': path}
        raw_paths = ingest_data(self.config)
        
        if not raw_paths:
            logging.error("La ingesta no retorno archivos validos.")
            raise FileNotFoundError("Ingestion failed")
            
        return raw_paths

    def run_validation(self, raw_paths):
        """Paso 2: Validacion"""
        if not self.config["validation"]["enabled"]:
            logging.info("Validacion desactivada.")
            return True

        logging.info(">>> PASO 2: VALIDACION DE DATOS CRUDOS")
        # validate_raw_data retorna True/False y loguea los detalles
        is_valid = validate_raw_data(raw_paths, self.config)
        
        if not is_valid:
            # En DataOps estricto, esto deberia detener el pipeline
            # Aqui permitimos continuar si es solo warning, segun logica de validation.py
            logging.warning("La validacion reporto problemas, pero continuando pipeline...")
        
        return is_valid

    def run_transformation(self, raw_paths):
        """Paso 3: Transformacion"""
        if not self.config["transformation"]["enabled"]:
            logging.info("Transformacion desactivada.")
            return None

        logging.info(">>> PASO 3: TRANSFORMACION Y LIMPIEZA")
        # transform_data retorna un DataFrame unificado
        df = transform_data(raw_paths, self.config)
        
        if df is None or df.empty:
            raise ValueError("La transformacion resulto en un DataFrame vacio")
            
        return df

    def run_nlp_processing(self, df):
        """Paso 4: NLP"""
        if not self.config["nlp"]["enabled"]:
            logging.info("NLP desactivado.")
            return df, {}

        logging.info(">>> PASO 4: PROCESAMIENTO NLP (Sentiment & Topics)")
        # nlp.py retorna tupla: (df_enriquecido, diccionario_resultados)
        df_processed, nlp_results = run_nlp_pipeline(df, self.config)
        
        return df_processed, nlp_results

    def run_network_analysis(self, df):
        """Paso 5: Analisis de Redes"""
        if not self.config["network_analysis"]["enabled"]:
            logging.info("Analisis de redes desactivado.")
            return {}

        logging.info(">>> PASO 5: ANALISIS DE REDES (Influencers)")
        # Retorna diccionario de metricas y paths de imagenes
        network_results = run_network_analysis(df, self.config)
        return network_results

    def run_statistical_analysis(self, df, network_results):
        """Paso 6: Estadistica Avanzada"""
        if not self.config["analysis"]["enabled"]:
            logging.info("Analisis estadistico desactivado.")
            return {}

        logging.info(">>> PASO 6: ANALISIS ESTADISTICO (Correlaciones & Tests)")
        # analysis.py ahora maneja la logica de seleccion automatica de tests
        stats_results = run_statistical_analysis(df, network_results, self.config)
        return stats_results

    def save_final_results(self, df, nlp_results, network_results, stats_results, output_folder):
        """Guardado centralizado de artefactos"""
        logging.info(">>> PASO 7: GUARDADO DE RESULTADOS")
        
        # 1. Guardar Datos Procesados (Parquet)
        if df is not None:
            parquet_path = os.path.join(output_folder, "final_processed_data.parquet")
            try:
                df.to_parquet(parquet_path, index=False)
                logging.info(f"Dataset final guardado: {parquet_path}")
            except Exception as e:
                logging.error(f"Error guardando parquet: {e}")

        # 2. Guardar Reportes JSON
        reports = {
            "nlp_analysis.json": nlp_results,
            "network_analysis.json": network_results,
            "statistical_analysis.json": stats_results
        }

        for filename, data in reports.items():
            if data:
                path = os.path.join(output_folder, filename)
                try:
                    with open(path, 'w', encoding='utf-8') as f: 
                        json.dump(data, f, indent=4, default=str)
                    logging.info(f"Reporte generado: {path}")
                except Exception as e:
                    logging.error(f"Error guardando reporte {filename}: {e}")

    def run(self):
        """Ejecucion principal del Pipeline"""
        logging.info("===============================================")
        logging.info(f" INICIANDO PIPELINE - Run ID: {self.current_run_id}")
        logging.info("===============================================")

        try:
            # 1. Setup
            output_folder = self.create_output_folders()

            # 2. Ingesta
            raw_paths = self.run_ingestion()
            
            # 3. Validacion
            self.run_validation(raw_paths)
            
            # 4. Transformacion
            df_main = self.run_transformation(raw_paths)
            
            # 5. NLP (Enriquece el DF)
            df_enriched, nlp_results = self.run_nlp_processing(df_main)
            
            # 6. Network Analysis (Usa DF enriquecido)
            network_results = self.run_network_analysis(df_enriched)
            
            # 7. Statistical Analysis (Core del proyecto)
            stats_results = self.run_statistical_analysis(df_enriched, network_results)
            
            # 8. Guardado y Reporte
            self.save_final_results(
                df_enriched, 
                nlp_results, 
                network_results, 
                stats_results, 
                output_folder
            )

            logging.info("===============================================")
            logging.info(" PIPELINE COMPLETADO EXITOSAMENTE ")
            logging.info(f" Resultados disponibles en: {output_folder}")
            logging.info("===============================================")
            
            return True

        except Exception as e:
            logging.critical("!ERROR CRITICO - EL PIPELINE SE DETUVO!")
            logging.critical(f"Motivo: {str(e)}")
            import traceback
            logging.critical(traceback.format_exc())
            raise e