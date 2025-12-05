import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

np.random.seed(42)

def create_twitter_mock_data():
    """Genera datos simulados de Twitter con estadística realista"""
    print("Generating Twitter Mock Data...")
    
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]  
    twitter_data = []
    user_counter = 1000 
    
    queries = ['AI', 'Machine Learning', 'Data Science', 'Cloud Computing', 'Big Data']
    
    for i in range(200):
        base_engagement = int(np.random.lognormal(mean=2.5, sigma=1.2))
        
        tweet = {
            'id': f"tw_{i}",
            'created_at': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
            'text': f"Tech tweet about {np.random.choice(queries)} #{i}. Innovation is key!",
            'user_id': f"user_{user_counter}",  
            'user_screen_name': f"tech_enthusiast_{user_counter}",
            'user_followers_count': int(np.random.pareto(a=1.16) * 1000) + 10,
            'retweet_count': int(base_engagement * 0.3),
            'favorite_count': int(base_engagement * 0.7),
            'reply_count': int(base_engagement * 0.1), 
            'lang': 'en',
            'source': 'twitter',
            'data_source': 'twitter'
        }
        twitter_data.append(tweet)
        user_counter += 1  
    
    df_twitter = pd.DataFrame(twitter_data)
    
    if len(df_twitter) < 100:
        pass 
    
    df_twitter['twitter_engagement'] = df_twitter['retweet_count'] + df_twitter['favorite_count']
    
    return df_twitter

def create_kaggle_fallback_mock():
    """
    CRÍTICO PARA CI/CD:
    Genera un archivo 'kaggle_reddit.csv' falso con la estructura REAL del dataset de Kaggle.
    Esto permite que el pipeline corra en GitHub Actions sin descargar gigabytes.
    """
    print("Generating Kaggle/Reddit Fallback Data (Real Schema)...")
    
    reddit_data = []
    start_date = datetime(2023, 1, 1)
    
    for i in range(200): 
        post = {
            'post_id': f"rd_{i}",
            'datetime': (start_date + timedelta(days=np.random.randint(0, 365))).strftime("%Y-%m-%d %H:%M:%S"),
            'title': f"Discussion about {np.random.choice(['AI', 'Tech', 'Coding'])}",
            'text': f"This is a sample text content regarding technology trends {i}. It is very interesting.",
            'score': np.random.randint(1, 500),
            'comments': np.random.randint(0, 50),
            'tag': np.random.choice(['artificial', 'MachineLearning', 'technology']),
            'author_post_karma': np.random.randint(0, 10000), 
            'data_source': 'reddit'
        }
        reddit_data.append(post)
    
    return pd.DataFrame(reddit_data)

def save_mock_data():
    """Guardar todos los mocks en data/raw"""
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- Creating Data Fixtures in {output_dir} ---")
    
    df_tw = create_twitter_mock_data()
    df_tw.to_csv(os.path.join(output_dir, 'twitter_mock_data.csv'), index=False)
    
    df_rd = create_kaggle_fallback_mock()
    df_rd.to_csv(os.path.join(output_dir, 'kaggle_reddit.csv'), index=False)
    
    df_rd.to_csv(os.path.join(output_dir, 'reddit_mock_data.csv'), index=False)
    
    print(f" Twitter Generated: {len(df_tw)} rows")
    print(f" Kaggle/Reddit Fallback Generated: {len(df_rd)} rows")

if __name__ == "__main__":
    save_mock_data()