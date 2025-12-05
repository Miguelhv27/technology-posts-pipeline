import os
import logging
import pandas as pd
import kagglehub
import shutil
from datetime import datetime

class DataIngestion:
    """
    DataOps Ingestion Engine.
    Estrategia: Intenta API de Kaggle -> Si falla (403/Red), usa archivo local de respaldo.
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

    def download_kaggle_dataset(self):
        """
        Intenta descargar via API. Si falla, hace fallback a archivo local manual.
        """
        if not self.data_sources["reddit"]["enabled"]:
            return None

        logging.info("---  Iniciando Ingesta: Reddit (Kaggle) ---")
        
        # Variable para guardar la ruta final del archivo
        final_csv_path = None

        # 1. INTENTO API
        try:
            if self._verify_kaggle_auth():
                dataset_name = self.data_sources["reddit"]["kaggle_dataset"]
                logging.info(f"   Intentando conexión API a: {dataset_name}...")
                
                path = kagglehub.dataset_download(dataset_name)
                
                # Buscar CSV descargado
                csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
                if csv_files:
                    final_csv_path = os.path.join(path, csv_files[0])
                    logging.info("    Descarga API exitosa.")
            else:
                logging.warning("    No se detectó token kaggle.json. Saltando intento API.")

        except Exception as e:
            logging.warning(f"    La API de Kaggle falló ({str(e)}).")
            logging.info("    Activando protocolo de respaldo (Fallback)...")

        if final_csv_path is None:
            manual_path = os.path.join(self.raw_data_dir, "kaggle_reddit.csv")
            if os.path.exists(manual_path):
                logging.info(f"    Archivo de respaldo local encontrado: {manual_path}")
                final_csv_path = manual_path
            else:
                logging.critical("    FALLO TOTAL: No funcionó la API y no existe 'data/raw/kaggle_reddit.csv'")
                return None


        try:
            logging.info("   Procesando dataset...")
            df = pd.read_csv(final_csv_path)
            
            sample_size = min(30000, len(df))
            sampled_df = df.sample(n=sample_size, random_state=42)
            sampled_df['data_source'] = 'reddit'
            
            output_path = os.path.join(self.raw_data_dir, "kaggle_sampled.csv")
            sampled_df.to_csv(output_path, index=False)
            
            logging.info(f"    Dataset Reddit listo: {len(sampled_df)} registros -> {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"    Error procesando el archivo CSV: {e}")
            return None

    def fetch_twitter_data(self):
        """Ingesta Simulada de Twitter"""
        try:
            if not self.data_sources["twitter"]["enabled"]:
                return None

            logging.info("---  Iniciando Ingesta API: Twitter (Simulada) ---")
            twitter_path = self.data_sources["twitter"].get("mock_data_path", "data/raw/twitter_mock_data.csv")
            
            if os.path.exists(twitter_path):
                logging.info(f"    Respuesta API simulada recibida: {twitter_path}")
                return twitter_path
            else:
                logging.warning(f"    No se encontró mock de Twitter.")
                return None
                
        except Exception as e:
            logging.error(f"Error Twitter Ingestion: {e}")
            return None

    def run(self):
        if not self.config.get("ingestion", {}).get("enabled", True):
            logging.info("Ingesta desactivada en configuración. Saltando paso.")
            return {}

        results = {}
        
        reddit_file = self.download_kaggle_dataset()
        if reddit_file:
            results['reddit_kaggle'] = reddit_file
        else:
            raise FileNotFoundError("Ingestión de Reddit falló (API y Local)")
        
        twitter_file = self.fetch_twitter_data()
        if twitter_file:
            results['twitter'] = twitter_file
            
        return results

def ingest_data(config):
    ingestor = DataIngestion(config)
    return ingestor.run()