import os
import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.transformation_config = config["transformation"]
        self.raw_data_dir = config["paths"]["raw_data_dir"]
        self.processed_data_dir = config["paths"]["processed_data_dir"]
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_and_combine_data(self, raw_paths):
        """Load and combine data from multiple sources"""
        combined_data = []
        
        for source_type, file_path in raw_paths.items():
            if file_path is None:
                continue
                
            if isinstance(file_path, list):
                for single_file in file_path:
                    self._load_single_file(single_file, source_type, combined_data)
            else:
                self._load_single_file(file_path, source_type, combined_data)
        
        if not combined_data:
            raise ValueError("No se pudieron cargar datos de ninguna fuente")
            
        return pd.concat(combined_data, ignore_index=True)

    def _load_single_file(self, file_path, source_type, combined_data):
        """Load a single data file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                logging.warning(f"Formato no soportado: {file_path}")
                return
            
            df['data_source'] = source_type
            df['ingestion_timestamp'] = datetime.now()
            
            combined_data.append(df)
            logging.info(f"Cargados {len(df)} registros de {file_path}")
            
        except Exception as e:
            logging.error(f"Error cargando {file_path}: {e}")

    def clean_text(self, text):
        """Apply comprehensive text cleaning"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        rules = self.transformation_config["text_cleaning"]
        
        if rules.get("lowercase", True):
            text = text.lower()
        
        if rules.get("remove_urls", True):
            text = re.sub(r"http\S+|www\S+", "", text)
        
        if rules.get("remove_html", True):
            text = re.sub(r"<.*?>", "", text)
        
        if rules.get("remove_special_chars", True):
            text = re.sub(r"[^a-zA-Z0-9áéíóúñü¿?¡!.,;:\s]", "", text)
        
        if rules.get("remove_punctuation", False):
            text = re.sub(r"[^\w\s]", "", text)
        
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    
    def extract_features(self, df):
        """Extract engineered features based on configuration"""
        features_config = self.transformation_config["feature_engineering"]
        df_features = df.copy()
        
        if 'text_cleaned' in df_features.columns:
            df_features['word_count'] = df_features['text_cleaned'].str.split().str.len().fillna(0)
        
        if 'score' in df_features.columns:
            if 'num_comments' in df_features.columns:
                df_features['reddit_engagement'] = df_features['score'] + (df_features['num_comments'] * 2)  
                df_features['reddit_virality'] = df_features['score'] * np.log1p(df_features['num_comments'])
            elif 'comments' in df_features.columns:
                df_features['reddit_engagement'] = df_features['score'] + (df_features['comments'] * 2)  
                df_features['reddit_virality'] = df_features['score'] * np.log1p(df_features['comments'])
            else:
                df_features['reddit_engagement'] = df_features['score']
                df_features['reddit_virality'] = df_features['score']
        
        if all(col in df_features.columns for col in ['retweet_count', 'favorite_count']):
            df_features['twitter_engagement'] = df_features['retweet_count'] + df_features['favorite_count']
        
        engagement_cols = []
        if 'reddit_engagement' in df_features.columns:
            engagement_cols.append('reddit_engagement')
        if 'twitter_engagement' in df_features.columns:
            engagement_cols.append('twitter_engagement')
        
        if engagement_cols:
            df_features['total_engagement'] = df_features[engagement_cols].sum(axis=1)
        
        df_features['engagement_rate'] = 0.0  
        
        if 'reddit_engagement' in df_features.columns:
            reddit_mask = df_features['data_source'].str.contains('reddit', na=False)
            if 'score' in df_features.columns:
                score_replace = df_features['score'].replace(0, 1)
                df_features.loc[reddit_mask, 'engagement_rate'] = (
                    df_features.loc[reddit_mask, 'reddit_engagement'] / score_replace[reddit_mask]
                ).fillna(0)
            else:
                engagement_replace = df_features['reddit_engagement'].replace(0, 1)
                df_features.loc[reddit_mask, 'engagement_rate'] = (
                    df_features.loc[reddit_mask, 'reddit_engagement'] / engagement_replace[reddit_mask]
                ).fillna(0)
        
        if 'twitter_engagement' in df_features.columns:
            twitter_mask = df_features['data_source'] == 'twitter'
            if 'retweet_count' in df_features.columns:
                retweet_replace = df_features['retweet_count'].replace(0, 1)
                df_features.loc[twitter_mask, 'engagement_rate'] = (
                    df_features.loc[twitter_mask, 'twitter_engagement'] / retweet_replace[twitter_mask]
                ).fillna(0)
            else:
                engagement_replace = df_features['twitter_engagement'].replace(0, 1)
                df_features.loc[twitter_mask, 'engagement_rate'] = (
                    df_features.loc[twitter_mask, 'twitter_engagement'] / engagement_replace[twitter_mask]
                ).fillna(0)
        
        if 'user_followers_count' not in df_features.columns:
            if 'author_post_karma' in df_features.columns:
                df_features['user_followers_count'] = df_features['author_post_karma']
            else:
                df_features['user_followers_count'] = 100
        
        return df_features


    def apply_text_cleaning(self, df):
        """Apply text cleaning to relevant columns"""
        text_columns = []
        
        for col in df.columns:
            if col in ['text', 'title', 'selftext', 'body', 'content']:
                text_columns.append(col)
        
        for col in text_columns:
            if col in df.columns:
                df[f'{col}_cleaned'] = df[col].apply(self.clean_text)
                logging.info(f"Aplicada limpieza de texto a columna: {col}")
        
        return df

    def calculate_sentiment(self, df):
        """Calculate sentiment scores using VADER"""
        if not self.transformation_config["feature_engineering"].get("sentiment", True):
            return df
        
        analyzer = SentimentIntensityAnalyzer()
        
        text_columns = [col for col in df.columns if col.endswith('_cleaned')]
        
        for text_col in text_columns:
            base_col = text_col.replace('_cleaned', '')
            df[f'{base_col}_sentiment'] = df[text_col].apply(
                lambda x: analyzer.polarity_scores(str(x))['compound'] if pd.notna(x) else 0.0
            )
            
            df[f'{base_col}_sentiment_label'] = df[f'{base_col}_sentiment'].apply(
                lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral'
            )
        
        logging.info("Análisis de sentimiento completado")
        return df

    def handle_missing_values(self, df):
        """Handle missing values based on data type"""
        df_fixed = df.copy()
        
        for col in df_fixed.columns:
            if df_fixed[col].dtype == 'object':
                df_fixed[col] = df_fixed[col].fillna('')
            elif np.issubdtype(df_fixed[col].dtype, np.number):
                df_fixed[col] = df_fixed[col].fillna(0)
        
        return df_fixed

    def normalize_numeric_features(self, df):
        """Normalize numeric features for analysis"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        exclude_cols = ['id', 'score', 'num_comments', 'retweet_count', 'favorite_count']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            if df[col].std() > 0:
                df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
        
        return df

    def run_transformation(self, raw_paths):
        """Execute complete transformation pipeline"""
        logging.info("Iniciando transformación de datos")
        
        if not self.transformation_config["enabled"]:
            logging.info("Transformación desactivada en configuración")
            return None

        try:
            df = self.load_and_combine_data(raw_paths)
            logging.info(f"Datos combinados: {len(df)} registros")
            
            df = self.apply_text_cleaning(df)
            
            df = self.handle_missing_values(df)
            
            df = self.extract_features(df)
            
            df = self.calculate_sentiment(df)
            
            df = self.normalize_numeric_features(df)
            
            output_file = os.path.join(self.processed_data_dir, "processed_data.parquet")
            df.to_parquet(output_file, index=False)
            logging.info(f"Datos transformados guardados en: {output_file}")
            
            self.log_transformation_stats(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error en transformación de datos: {e}")
            raise

    def log_transformation_stats(self, df):
        """Log transformation statistics"""
        stats = {
            'total_records': len(df),
            'sources': df['data_source'].value_counts().to_dict(),
            'columns_final': list(df.columns),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        logging.info(f"Estadísticas de transformación: {stats}")

    def run(self, raw_paths):
        return self.run_transformation(raw_paths)

def transform_data(raw_paths, config):
    """Main transformation function"""
    transformer = DataTransformation(config)
    return transformer.run(raw_paths)