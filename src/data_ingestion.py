import os
import logging
import pandas as pd
import kagglehub
import requests
from datetime import datetime
import time
from datetime import timedelta
import pandas as pd  
import numpy as np

class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.raw_data_dir = config["paths"]["raw_data_dir"]
        self.data_sources = config["data_sources"]
        os.makedirs(self.raw_data_dir, exist_ok=True)

    def download_kaggle_dataset(self):
        """Download and sample Kaggle dataset"""
        try:
            dataset_name = self.data_sources["reddit"]["kaggle_dataset"]
            logging.info(f"Downloading Kaggle dataset: {dataset_name}")
            
            path = kagglehub.dataset_download(dataset_name)
            
            csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
            if csv_files:
                full_path = os.path.join(path, csv_files[0])
                df = pd.read_csv(full_path)
                
                sample_size = min(10000, len(df) // 100)
                sampled_df = df.sample(n=sample_size, random_state=42)
                
                sample_path = os.path.join(self.raw_data_dir, "kaggle_sampled.csv")
                sampled_df.to_csv(sample_path, index=False)
                
                logging.info(f"Kaggle dataset sampled: {len(sampled_df)} records")
                return sample_path
                
        except Exception as e:
            logging.error(f"Kaggle download failed: {e}")
            return None

    def fetch_twitter_data(self):
        """Fetch data from Twitter API"""
        try:
            if not self.data_sources["twitter"]["enabled"]:
                logging.info("Twitter API desactivada en configuracion")
                return None

            logging.info("Iniciando recoleccion de datos de Twitter")
            
            twitter_data = []
            for query in self.data_sources["twitter"]["search_queries"]:
                mock_tweet = {
                    'id': f"tw_{int(time.time())}_{len(twitter_data)}",
                    'created_at': datetime.now().isoformat(),
                    'text': f"Ejemplo de tweet sobre {query}",
                    'user_id': f"user_{len(twitter_data)}",
                    'retweet_count': len(twitter_data) * 2,
                    'favorite_count': len(twitter_data) * 3,
                    'query': query,
                    'source': 'twitter'
                }
                twitter_data.append(mock_tweet)
            
            df_twitter = pd.DataFrame(twitter_data)
            twitter_file = os.path.join(self.raw_data_dir, "twitter_data.csv")
            df_twitter.to_csv(twitter_file, index=False)
            logging.info(f"Datos de Twitter guardados: {twitter_file}")
            
            return twitter_file
            
        except Exception as e:
            logging.error(f"Error obteniendo datos de Twitter: {e}")
            return None

    def fetch_reddit_data(self):
        """Fetch data from Reddit API"""
        try:
            if not self.data_sources["reddit"]["enabled"]:
                logging.info("Reddit API desactivada en configuracion")
                return None

            logging.info("Generando datos mock de Reddit con mayor volumen")
            
            reddit_data = []
            subreddits = self.data_sources["reddit"]["subreddits"]
            
            for subreddit in subreddits:
                for post_num in range(150): 
                    post = {
                        'id': f"rd_{int(time.time())}_{subreddit}_{post_num}",
                        'created_utc': int((datetime.now() - timedelta(days=post_num % 30)).timestamp()),
                        'title': f"Technology discussion about {np.random.choice(['AI', 'Machine Learning', 'Data Science', 'Cloud'])} in {subreddit}",
                        'selftext': f"This is a comprehensive discussion about {np.random.choice(['artificial intelligence', 'machine learning algorithms', 'data engineering', 'cloud computing'])}. Many users are sharing their insights and experiences with this technology.",
                        'score': np.random.randint(10, 1000),
                        'num_comments': np.random.randint(5, 100),
                        'upvote_ratio': np.random.uniform(0.7, 0.95),
                        'subreddit': subreddit,
                        'author': f"reddit_user_{np.random.randint(1000, 9999)}",
                        'source': 'reddit'
                    }
                    reddit_data.append(post)
            
            df_reddit = pd.DataFrame(reddit_data)
            reddit_file = os.path.join(self.raw_data_dir, "reddit_generated_data.csv")
            df_reddit.to_csv(reddit_file, index=False)
            logging.info(f"Datos de Reddit generados: {reddit_file} - {len(df_reddit)} registros")
            
            return reddit_file
            
        except Exception as e:
            logging.error(f"Error generando datos de Reddit: {e}")
            return None

    def load_kaggle_data(self, downloaded_path):
        """Load and process Kaggle dataset files"""
        try:
            if not downloaded_path:
                return None

            logging.info(f"Cargando archivos del dataset desde {downloaded_path}")
            csv_files = [f for f in os.listdir(downloaded_path) if f.endswith(".csv")]

            if not csv_files:
                logging.warning("No se encontraron archivos CSV en el dataset descargado")
                return None

            saved_files = []
            for file in csv_files:
                full_path = os.path.join(downloaded_path, file)
                df = pd.read_csv(full_path)
                
                required_columns = self.config["validation"]["required_columns"]["reddit"]
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = None 
                
                save_path = os.path.join(self.raw_data_dir, f"kaggle_{file}")
                df.to_csv(save_path, index=False)
                saved_files.append(save_path)
                logging.info(f"Dataset de Kaggle guardado: {save_path}")

            return saved_files
            
        except Exception as e:
            logging.error(f"Error procesando datos de Kaggle: {e}")
            return None
        

    def load_existing_kaggle_data(self):
        """Load existing Kaggle data from data/raw/kaggle_reddit_data/data.csv"""
        try:
            kaggle_path = "data/raw/kaggle_reddit_data/data.csv"
            
            if os.path.exists(kaggle_path):
                logging.info(f"Loading existing Kaggle data from: {kaggle_path}")
                
                df = pd.read_csv(kaggle_path)
                logging.info(f"Loaded {len(df)} records from Kaggle dataset")
                
                sample_size = min(30000, len(df) // 100)
                sampled_df = df.sample(n=sample_size, random_state=42)
                logging.info(f"Sampled to {len(sampled_df)} records for development")
                
                sample_path = os.path.join(self.raw_data_dir, "kaggle_sampled.csv")
                sampled_df.to_csv(sample_path, index=False)
                logging.info(f"Saved sampled data to: {sample_path}")
                
                return sample_path
            else:
                logging.warning(f"Kaggle data not found at: {kaggle_path}")
                return None
                
        except Exception as e:
            logging.error(f"Error loading existing Kaggle data: {e}")
            return None

    def validate_api_credentials(self):
        """Validate that API credentials are available"""
        missing_creds = []
        
        if self.data_sources["twitter"]["enabled"]:
            twitter_creds = self.config["apis"]["twitter"]
            for key, value in twitter_creds.items():
                if value.startswith("${") and value.endswith("}"):
                    missing_creds.append(f"Twitter {key}")
        
        if self.data_sources["reddit"]["enabled"]:
            reddit_creds = self.config["apis"]["reddit"]
            for key, value in reddit_creds.items():
                if key in ["client_id", "client_secret"] and value.startswith("${") and value.endswith("}"):
                    missing_creds.append(f"Reddit {key}")
        
        if missing_creds:
            logging.warning(f"Credenciales faltantes: {', '.join(missing_creds)}")
            return False
        return True

    def load_mock_data(self):
        """Load mock data from local files"""
        mock_data_paths = {}
        
        try:
            if self.data_sources["twitter"]["enabled"]:
                twitter_path = self.data_sources["twitter"].get("mock_data_path", "data/raw/twitter_mock_data.csv")
                if os.path.exists(twitter_path):
                    mock_data_paths['twitter'] = twitter_path
                    logging.info(f"Loaded Twitter mock data: {twitter_path}")
            
            if self.data_sources["reddit"]["enabled"]:
                reddit_path = self.data_sources["reddit"].get("mock_data_path", "data/raw/reddit_mock_data.csv")
                if os.path.exists(reddit_path):
                    mock_data_paths['reddit_api'] = reddit_path
                    logging.info(f"Loaded Reddit mock data: {reddit_path}")
                else:
                    # Si no existe, generar datos Reddit
                    logging.info("Reddit mock data not found, generating...")
                    reddit_file = self.fetch_reddit_data()
                    if reddit_file:
                        mock_data_paths['reddit_api'] = reddit_file
                        
        except Exception as e:
            logging.error(f"Error loading mock data: {e}")
        
        return mock_data_paths

    def load_reddit_mock_fallback(self):
        """Cargar datos mock de Reddit como fallback si Kaggle falla"""
        reddit_mock_path = self.data_sources["reddit"].get("mock_data_path")
        if reddit_mock_path and os.path.exists(reddit_mock_path):
            logging.info(f"Cargando datos mock de Reddit como fallback: {reddit_mock_path}")
            return reddit_mock_path
        return None

    def generate_development_data(self):
        """Generate development data when no mock files exist"""
        try:
            results = {}
            
            if self.data_sources["twitter"]["enabled"]:
                twitter_file = self.fetch_twitter_data()
                if twitter_file:
                    results['twitter'] = twitter_file
                    logging.info(f"Generated Twitter data: {twitter_file}")
            
            if self.data_sources["reddit"]["enabled"]:
                reddit_file = self.fetch_reddit_data()
                if reddit_file:
                    results['reddit_api'] = reddit_file
                    logging.info(f"Generated Reddit data: {reddit_file}")
            
            commercial_data = self.create_commercial_data()
            if commercial_data:
                results['commercial'] = commercial_data
                logging.info(f"Generated commercial data")
            
            logging.info(f" Desarrollo data generation completed: {list(results.keys())}")
            return results
            
        except Exception as e:
            logging.error(f"Development data generation failed: {e}")
            return {}
        
    def create_commercial_data(self):
        """Create commercial data for correlation analysis"""
        try:
            import pandas as pd
            from datetime import datetime, timedelta
            import numpy as np
            
            dates = [datetime.now() - timedelta(days=x) for x in range(30)]
            commercial_data = []
            
            for date in dates:
                commercial_data.append({
                    'date': date.date(),
                    'amazon_sales_velocity': np.random.randint(1000, 5000),
                    'sephora_review_volume': np.random.randint(500, 2000),
                    'restaurant_daily_sales': np.random.randint(2000, 8000),
                    'steam_game_purchases': np.random.randint(100, 1000),
                    'supplement_sales': np.random.randint(300, 1500)
                })
            
            df_commercial = pd.DataFrame(commercial_data)
            commercial_path = os.path.join(self.raw_data_dir, "commercial_data.csv")
            df_commercial.to_csv(commercial_path, index=False)
            logging.info(f"Commercial data saved: {commercial_path}")
            
            return commercial_path
            
        except Exception as e:
            logging.error(f"Commercial data creation failed: {e}")
            return None
        

    def run(self):
        """Execute the complete ingestion process"""
        logging.info("Iniciando proceso de ingesta de datos")
        
        if not self.config["ingestion"]["enabled"]:
            logging.info("Ingesta desactivada en configuracion")
            return {}

        results = {}
        
        logging.info(" Forcing REAL Kaggle data for Reddit...")
        kaggle_file = self.load_existing_kaggle_data()
        if kaggle_file:
            results['reddit_kaggle'] = kaggle_file
            logging.info(" REAL Kaggle data loaded successfully")
        else:
            logging.error(" Failed to load Kaggle data, but continuing with other sources")
        
        logging.info(" Forcing Twitter mock data...")
        twitter_mock_path = self.data_sources["twitter"].get("mock_data_path", "data/raw/twitter_mock_data.csv")
        
        if os.path.exists(twitter_mock_path):
            results['twitter'] = twitter_mock_path
            logging.info(f" Twitter mock data loaded: {twitter_mock_path}")
            
            try:
                df_twitter = pd.read_csv(twitter_mock_path)
                logging.info(f" Twitter mock data contains {len(df_twitter):,} records")
            except Exception as e:
                logging.warning(f"Could not verify Twitter data size: {e}")
        else:
            logging.warning(" Twitter mock data not found, generating...")
            twitter_file = self.fetch_twitter_data()
            if twitter_file:
                results['twitter'] = twitter_file
                logging.info(" Generated Twitter data as fallback")
        
        has_reddit = any('reddit' in key for key in results.keys())
        has_twitter = any('twitter' in key for key in results.keys())
        
        if has_reddit and has_twitter:
            logging.info(" SUCCESS: Both Reddit (Kaggle) and Twitter data loaded!")
        elif has_reddit:
            logging.warning("  Only Reddit data loaded - missing Twitter")
        elif has_twitter:
            logging.warning("  Only Twitter data loaded - missing Reddit")
        else:
            logging.error(" CRITICAL: No data sources loaded!")
            
            logging.info(" Generating emergency fallback data...")
            twitter_file = self.fetch_twitter_data()
            reddit_file = self.fetch_reddit_data()
            
            if twitter_file:
                results['twitter'] = twitter_file
            if reddit_file:
                results['reddit_api'] = reddit_file

        successful_ingestions = {k: v for k, v in results.items() if v is not None}
        
        logging.info(f" Proceso de ingesta completado. Fuentes exitosas: {list(successful_ingestions.keys())}")
        
        for source, path in successful_ingestions.items():
            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                    logging.info(f"   {source}: {len(df):,} records")
                elif isinstance(path, list):
                    total_records = 0
                    for p in path:
                        if os.path.exists(p):
                            df = pd.read_csv(p)
                            total_records += len(df)
                    logging.info(f"   {source}: {total_records:,} records (multiple files)")
            except Exception as e:
                logging.warning(f"   Could not count records for {source}: {e}")
        
        return successful_ingestions
        

def ingest_data(config):
    """Main function to run data ingestion"""
    ingestion = DataIngestion(config)
    return ingestion.run()