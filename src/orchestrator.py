import os
import yaml
import logging
from datetime import datetime
import json

from src.data_ingestion import ingest_data
from src.data_validation import validate_raw_data
from src.data_transformation import transform_data
from src.nlp import run_nlp_pipeline
from src.analysis import run_statistical_analysis
from src.network_analysis import run_network_analysis

def configure_logging(log_dir="logs/"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

class Orchestrator:

    def __init__(self, config_path="config/pipeline_config.yaml"):
        self.config = self.load_config(config_path)
        self.processed_data_dir = self.config["paths"]["processed_data_dir"]
        self.raw_data_dir = self.config["paths"]["raw_data_dir"]
        self.logs_dir = self.config["paths"]["logs_dir"]
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def create_versioned_output_folder(self):
        if self.config["pipeline"]["create_timestamped_folders"]:
            folder = os.path.join(self.processed_data_dir, self.current_run_id)
        else:
            folder = self.processed_data_dir
            
        os.makedirs(folder, exist_ok=True)
        logging.info(f"Carpeta de salida creada: {folder}")
        return folder

    def initialize_environment(self):
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        if self.config["pipeline"]["versioning"]:
            logging.info("Sistema de versionamiento activado")

    def run_ingestion(self):
        if not self.config["ingestion"]["enabled"]:
            logging.info("Ingesta desactivada por configuración")
            return self.raw_data_dir

        logging.info("Paso 1: Ingesta de datos desde multiples fuentes")
        raw_data_paths = ingest_data(self.config)
        return raw_data_paths

    def run_validation(self, raw_paths):
        if not self.config["validation"]["enabled"]:
            logging.info("Validación desactivada por configuración")
            return True

        logging.info("Paso 2: Validación de datos crudos")
        validation_passed = validate_raw_data(raw_paths, self.config)
        if not validation_passed:
            raise ValueError("Validación de datos falló")
        return validation_passed

    def run_transformation(self, raw_paths):
        if not self.config["transformation"]["enabled"]:
            logging.info("Transformación desactivada por configuración")
            return None

        logging.info("Paso 3: Transformación y feature engineering")
        transformed_data = transform_data(raw_paths, self.config)
        
        if self.config.get("development_mode", True):
            from src.data_sampling import sample_data_for_development
            transformed_data = sample_data_for_development(transformed_data, sample_fraction=0.2)
            logging.info("Datos muestreados para desarrollo (20% del total)")
        
        return transformed_data

    def run_nlp_processing(self, transformed_data):
        if not self.config["nlp"]["enabled"] or transformed_data is None:
            logging.info("Procesamiento NLP desactivado")
            return transformed_data

        logging.info("Paso 4: Procesamiento NLP (sentimiento y topic modeling)")
        nlp_processed_data, nlp_results = run_nlp_pipeline(transformed_data, self.config)
        return nlp_processed_data 

    def run_network_analysis(self, processed_data):
        if not self.config["network_analysis"]["enabled"] or processed_data is None:
            logging.info("Análisis de redes desactivado")
            return None

        logging.info("Paso 5: Análisis de redes e identificación de influencers")
        network_results = run_network_analysis(processed_data, self.config)
        return network_results

    def run_statistical_analysis(self, processed_data, network_results):
        if not self.config["analysis"]["enabled"] or processed_data is None:
            logging.info("Análisis estadístico desactivado")
            return None

        logging.info("Paso 6: Análisis estadístico y correlaciones")
        analysis_results = run_statistical_analysis(processed_data, network_results, self.config)
        return analysis_results

    def save_results(self, processed_data, analysis_results, network_results, output_folder):
        if processed_data is not None:
            output_file = os.path.join(output_folder, "processed_data.parquet")
            processed_data.to_parquet(output_file, compression=self.config["export"]["compression"])
            logging.info(f"Datos procesados guardados en: {output_file}")

        commercial_data_path = "data/raw/commercial_data.csv"
        if os.path.exists(commercial_data_path):
            import shutil
            shutil.copy2(commercial_data_path, os.path.join(output_folder, "commercial_data.csv"))
            logging.info(f"Datos comerciales copiados a: {output_folder}")

        if analysis_results:
            results_file = os.path.join(output_folder, "analysis_results.json")
            with open(results_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            logging.info(f"Resultados de análisis guardados en: {results_file}")

        if network_results:
            network_file = os.path.join(output_folder, "network_analysis.json")
            with open(network_file, 'w') as f:
                json.dump(network_results, f, indent=2, default=str)
            logging.info(f"Resultados de red guardados en: {network_file}")

    def check_alerts(self, analysis_results, network_results):
        if not self.config["alerts"]["sentiment_drop"]["enabled"]:
            return

        logging.info("Verificando sistema de alertas")

    def run(self):
        logging.info("===============================================")
        logging.info("INICIO DEL PIPELINE Technology Posts")
        logging.info("===============================================")

        try:
            self.initialize_environment()
            output_folder = self.create_versioned_output_folder()

            raw_paths = self.run_ingestion()
            
            self.run_validation(raw_paths)
            
            transformed_data = self.run_transformation(raw_paths)
            
            nlp_processed_data = self.run_nlp_processing(transformed_data)
            
            network_results = self.run_network_analysis(nlp_processed_data)
            
            analysis_results = self.run_statistical_analysis(nlp_processed_data, network_results)
            
            self.save_results(nlp_processed_data, analysis_results, network_results, output_folder)
            
            self.check_alerts(analysis_results, network_results)

            logging.info("===============================================")
            logging.info("PIPELINE COMPLETADO EXITOSAMENTE")
            logging.info("===============================================")

            return {
                "run_id": self.current_run_id,
                "raw_paths": raw_paths,
                "processed_path": output_folder,
                "analysis_results": analysis_results,
                "network_results": network_results
            }

        except Exception as e:
            logging.error("ERROR CRITICO EN EL PIPELINE")
            logging.error(f"Tipo de error: {type(e).__name__}")
            logging.error(f"Mensaje: {str(e)}")
            raise e

if __name__ == "__main__":
    configure_logging()
    orchestrator = Orchestrator()
    result = orchestrator.run()
