import os
import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.transformation_config = config["transformation"]
        self.raw_data_dir = config["paths"]["raw_data_dir"]
        self.processed_data_dir = config["paths"]["processed_data_dir"]
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def load_and_combine_data(self, raw_paths):
        combined_data = []
        for source_type, file_path in raw_paths.items():
            if not file_path: continue
            
            try:
                if file_path.endswith('.csv'): df = pd.read_csv(file_path)
                elif file_path.endswith('.parquet'): df = pd.read_parquet(file_path)
                else: continue
                
                if source_type == 'reddit':
                    rename_map = {
                        'post_id': 'id',
                        'datetime': 'created_at',
                        'tag': 'subreddit'
                    }
                    df.rename(columns=rename_map, inplace=True)
                    if 'subreddit' not in df.columns: df['subreddit'] = 'technology'
                    else: df['subreddit'] = df['subreddit'].fillna('technology')

                df['data_source'] = source_type
                df.columns = [c.lower().strip() for c in df.columns]
                combined_data.append(df)
                logging.info(f"Loaded {len(df)} records from {source_type}")
                
            except Exception as e:
                logging.error(f"Error loading {file_path}: {e}")

        if not combined_data: return pd.DataFrame()
        return pd.concat(combined_data, ignore_index=True)

    def clean_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"[^a-zA-Z0-9áéíóúñü¿?¡!.,;:\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def extract_features(self, df):
        """Feature Engineering - Removing dependency on unreliable karma/followers"""
        df_feat = df.copy()
        
        if 'text' in df_feat.columns:
            if 'title' in df_feat.columns:
                df_feat['text'] = df_feat['text'].fillna(df_feat['title'])
            df_feat['text_cleaned'] = df_feat['text'].apply(self.clean_text)
            df_feat['word_count'] = df_feat['text_cleaned'].str.split().str.len().fillna(0)
        
        if 'user_followers_count' in df_feat.columns:
            df_feat['user_followers_count'] = df_feat['user_followers_count'].fillna(0)
        elif 'author_post_karma' in df_feat.columns:
            df_feat['user_followers_count'] = df_feat['author_post_karma'].fillna(0)
        else:
            df_feat['user_followers_count'] = 0

        if 'retweet_count' in df_feat.columns:
            df_feat['twitter_engagement'] = df_feat['retweet_count'].fillna(0) + df_feat['favorite_count'].fillna(0)
        else:
            df_feat['twitter_engagement'] = 0
            
        if 'score' in df_feat.columns:
            comments = df_feat['comments'].fillna(0) if 'comments' in df_feat.columns else 0
            df_feat['reddit_engagement'] = df_feat['score'].fillna(0) + (comments * 2)
        else:
            df_feat['reddit_engagement'] = 0
            
        df_feat['total_engagement'] = df_feat['twitter_engagement'] + df_feat['reddit_engagement']

        max_eng = df_feat['total_engagement'].max()
        if max_eng > 0:
            df_feat['engagement_rate'] = df_feat['total_engagement'] / max_eng
        else:
            df_feat['engagement_rate'] = 0.0

        return df_feat

    def calculate_sentiment(self, df):
        if 'text_cleaned' not in df.columns: return df
        analyzer = SentimentIntensityAnalyzer()
        df['sentiment_compound'] = df['text_cleaned'].apply(
            lambda x: analyzer.polarity_scores(str(x))['compound'] if x else 0.0
        )
        df['sentiment_label'] = df['sentiment_compound'].apply(
            lambda x: 'positive' if x > 0.05 else 'negative' if x < -0.05 else 'neutral'
        )
        df['sentiment_polarity'] = df['sentiment_compound']
        return df

    def handle_missing_values(self, df):
        obj_cols = df.select_dtypes(include=['object']).columns
        df[obj_cols] = df[obj_cols].fillna('unknown')
        df['total_engagement'] = df['total_engagement'].fillna(0)
        if 'word_count' in df.columns:
            df['word_count'] = df['word_count'].fillna(0)
        return df

    def normalize_numeric_features(self, df):
        cols = ['total_engagement', 'word_count']
        for col in cols:
            if col in df.columns and df[col].std() > 0:
                df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()
        return df

    def run(self, raw_paths):
        if not self.transformation_config["enabled"]: return None
        try:
            df = self.load_and_combine_data(raw_paths)
            if df.empty: return df
            df = self.extract_features(df)
            df = self.calculate_sentiment(df)
            df = self.handle_missing_values(df)
            df = self.normalize_numeric_features(df)
            
            out_path = os.path.join(self.processed_data_dir, "processed_data.parquet")
            df.to_parquet(out_path, index=False)
            logging.info(f"Transformation saved: {len(df)} records")
            return df
        except Exception as e:
            logging.error(f"Transformation failed: {e}")
            raise

def transform_data(raw_paths, config):
    t = DataTransformation(config)
    return t.run(raw_paths)