import os
import logging
import pandas as pd
import kagglehub
import subprocess
import sys

class DataIngestion:
    """
    DataOps Ingestion Engine.
    Responsabilidad: Obtener datos de fuentes externas.
    Estrategia:
    1. Reddit: API Kaggle
    2. Twitter: Archivo Mock 
    """
    def __init__(self, config):
        self.config = config
        self.raw_data_dir = config["paths"]["raw_data_dir"]
        self.data_sources = config["data_sources"]
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def _verify_kaggle_auth(self):
        """Verifica existencia del token"""
        kaggle_dir = os.path.expanduser("~/.kaggle")
        token_path = os.path.join(kaggle_dir, "kaggle.json")
        return os.path.exists(token_path)

    def _trigger_external_mock_generation(self):
        """
        Simula una petición a un proveedor externo.
        Ejecuta el script scripts/create_mock_data.py como un subproceso.
        """
        try:
            logging.info("    Contactando proveedor de datos simulados (Ejecutando script externo)...")
            
            script_path = os.path.join("scripts", "create_mock_data.py")
            
            if not os.path.exists(script_path):
                logging.error(f"    Error: No se encuentra el generador en {script_path}")
                return False

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=True
            )
            
            logging.info("    Proveedor externo generó los datos exitosamente.")
            return True
            
        except subprocess.CalledProcessError as e:
            logging.error(f"    El generador externo falló: {e.stderr}")
            return False
        except Exception as e:
            logging.error(f"    Error invocando script externo: {e}")
            return False

    def download_kaggle_dataset(self):
        """Ingesta de Reddit (Kaggle)"""
        if not self.data_sources["reddit"]["enabled"]: return None
        
        logging.info("---  Iniciando Ingesta: Reddit ---")
        final_path = None

        try:
            if self._verify_kaggle_auth():
                dataset = self.data_sources["reddit"]["kaggle_dataset"]
                logging.info(f"   Intentando descarga API: {dataset}")
                path = kagglehub.dataset_download(dataset)
                csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
                if csvs:
                    final_path = os.path.join(path, csvs[0])
                    logging.info("    API Exitosa.")
        except Exception as e:
            logging.warning(f"    API falló: {e}")

        if not final_path:
            manual_path = os.path.join(self.raw_data_dir, "kaggle_reddit.csv")
            if os.path.exists(manual_path):
                logging.info(f"    Usando respaldo local: {manual_path}")
                final_path = manual_path

        if final_path:
            try:
                df = pd.read_csv(final_path)
                sample_size = min(30000, len(df))
                sampled = df.sample(n=sample_size, random_state=42)
                sampled['data_source'] = 'reddit_kaggle'
                
                out = os.path.join(self.raw_data_dir, "kaggle_sampled.csv")
                sampled.to_csv(out, index=False)
                return out
            except Exception as e:
                logging.error(f"Error procesando Reddit: {e}")
        
        logging.warning("    No se encontró Reddit Real. Solicitando Mock de Emergencia...")
        if self._trigger_external_mock_generation():
             fallback_mock = os.path.join(self.raw_data_dir, "kaggle_reddit.csv")
             if os.path.exists(fallback_mock):
                 return self._process_fallback_mock(fallback_mock)

        logging.error(" No se pudo cargar Reddit.")
        return None

    def _process_fallback_mock(self, path):
        """Helper para procesar el mock de reddit si fue generado"""
        df = pd.read_csv(path)
        df['data_source'] = 'reddit_kaggle'
        out = os.path.join(self.raw_data_dir, "kaggle_sampled.csv")
        df.to_csv(out, index=False)
        return out

    def fetch_twitter_data(self):
        """
        Ingesta de Twitter.
        Si el archivo no está, invoca al script externo para que lo provea.
        """
        if not self.data_sources["twitter"]["enabled"]: return None
        
        logging.info("---  Iniciando Ingesta: Twitter ---")
        twitter_path = self.data_sources["twitter"].get("mock_data_path", "data/raw/twitter_mock_data.csv")
        
        if os.path.exists(twitter_path):
            logging.info(f"    Datos recibidos de: {twitter_path}")
            return twitter_path

        logging.warning("    Fuente vacía. Solicitando regeneración a proveedor externo...")
        success = self._trigger_external_mock_generation()
        
        if success and os.path.exists(twitter_path):
            logging.info(f"    Datos recibidos exitosamente tras solicitud.")
            return twitter_path
        else:
            logging.error("    Fallo crítico: El proveedor externo no entregó los datos.")
            return None

    def run(self):
        if not self.config.get("ingestion", {}).get("enabled", True):
            logging.info("Ingesta desactivada en configuración. Saltando paso.")
            return {}

        results = {}
        
        rd = self.download_kaggle_dataset()
        if rd: results['reddit_kaggle'] = rd
        
        tw = self.fetch_twitter_data()
        if tw: results['twitter'] = tw
        
        return results

def ingest_data(config):
    ingestor = DataIngestion(config)
    return ingestor.run()