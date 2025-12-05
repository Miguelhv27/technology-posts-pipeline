import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

np.random.seed(42)

def create_twitter_mock_data():
    """
    Create mock Twitter data with realistic statistical distributions (Power Law/LogNormal).
    This ensures statistical tests in Analysis.py (Shapiro/Normaltest) work correctly.
    """
    print("Generating Twitter Mock Data...")
    
    dates = [datetime.now() - timedelta(days=x) for x in range(200)]  
    
    twitter_data = []
    user_counter = 1000 
    tweets_per_day = 30 
    
    queries = ['AI', 'Machine Learning', 'Data Science', 'Cloud Computing', 'Big Data', 
               'IoT', 'Blockchain', 'Neural Networks', 'Deep Learning', 'Computer Vision']
    
    for i, date in enumerate(dates):
        for j in range(tweets_per_day): 
            tweet_id = f"tw_{date.strftime('%Y%m%d')}_{j}"
            
            followers = int(np.random.pareto(a=1.16) * 1000) + 10
            
            base_engagement = int(np.random.lognormal(mean=2.5, sigma=1.2))
            
            retweet_count = int(base_engagement * 0.3)
            favorite_count = int(base_engagement * 0.7)
            
            tweet = {
                'id': tweet_id,
                'created_at': date.isoformat(),
                'text': f"Breaking news in {np.random.choice(queries)}! Revolutionary breakthrough regarding #{np.random.choice(['Tech', 'AI', 'Innovation'])}.",
                'user_id': f"user_{user_counter}",  
                'user_screen_name': f"tech_enthusiast_{user_counter}",
                'user_followers_count': followers, 
                'retweet_count': retweet_count,
                'favorite_count': favorite_count,
                'reply_count': int(base_engagement * 0.1), 
                'lang': 'en',
                'source': 'twitter',
                'data_source': 'twitter',
                'query': np.random.choice(queries)
            }
            twitter_data.append(tweet)
            user_counter += 1  
    
    df_twitter = pd.DataFrame(twitter_data)
    
    if len(df_twitter) < 6000:
        needed = 6000 - len(df_twitter)
        print(f"Adding {needed} filler records to reach volume target...")
        
        filler_data = []
        for k in range(needed):
            followers = int(np.random.pareto(a=1.16) * 1000) + 10
            base_eng = int(np.random.lognormal(mean=2.0, sigma=1.0))
            
            filler_data.append({
                'id': f"tw_fill_{k}",
                'created_at': datetime.now().isoformat(),
                'text': "Filler tech content for dataset balance.",
                'user_id': f"user_fill_{k}",
                'user_screen_name': f"filler_user_{k}",
                'user_followers_count': followers,
                'retweet_count': int(base_eng * 0.2),
                'favorite_count': int(base_eng * 0.8),
                'reply_count': 0,
                'lang': 'en',
                'source': 'twitter',
                'data_source': 'twitter',
                'query': 'general'
            })
        
        df_filler = pd.DataFrame(filler_data)
        df_twitter = pd.concat([df_twitter, df_filler], ignore_index=True)
    
    df_twitter['twitter_engagement'] = df_twitter['retweet_count'] + df_twitter['favorite_count']
    
    print(f"Twitter dataset generated: {len(df_twitter)} records")
    return df_twitter

def create_reddit_mock_data():
    """
    Create mock Reddit data.
    Note: Ideally we use Kaggle data, but this serves as a fallback 
    if the Kaggle download fails or for testing specific edge cases.
    """
    print("Generating Reddit Mock Data...")
    
    start_date = datetime(2023, 1, 1) 
    dates = [start_date + timedelta(days=x) for x in range(180)]  
    subreddits = ['artificial', 'MachineLearning', 'datascience', 'technology', 'programming']
    
    reddit_data = []
    author_counter = 10000
    posts_per_day = 5  
    
    for i, date in enumerate(dates):
        for subreddit in subreddits:
            for post_num in range(posts_per_day):  
                post_id = f"rd_{date.strftime('%Y%m%d')}_{subreddit}_{post_num}"
                
                score = int(np.random.lognormal(mean=3, sigma=1.5))
                
                post = {
                    'id': post_id,
                    'created_utc': int(date.timestamp()),  
                    'title': f"Discussion about {np.random.choice(['AI', 'ML', 'Data Engineering'])} in {subreddit}",
                    'text': f"Insightful discussion about {subreddit} trends. The community is debating the impact of recent algorithms.",
                    'score': score,  
                    'num_comments': int(score * np.random.uniform(0.1, 0.8)),  
                    'upvote_ratio': np.random.uniform(0.6, 0.99),
                    'subreddit': subreddit,
                    'author': f"user_{author_counter}",
                    'author_post_karma': int(np.random.pareto(a=1.5) * 5000) + 100,
                    'data_source': 'reddit'
                }
                reddit_data.append(post)
                author_counter += 1
    
    df_reddit = pd.DataFrame(reddit_data)
    print(f"Reddit dataset generated: {len(df_reddit)} records")
    return df_reddit

def save_mock_data():
    """Save mock data to raw data directory"""
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    
    print("--- Starting Mock Data Generation ---")
    
    df_twitter = create_twitter_mock_data()
    df_twitter.to_csv(os.path.join(output_dir, 'twitter_mock_data.csv'), index=False)

    df_reddit = create_reddit_mock_data()
    df_reddit.to_csv(os.path.join(output_dir, 'reddit_mock_data.csv'), index=False)
    
    print("\n--- Summary ---")
    print(f"Twitter Records: {len(df_twitter)}")
    print(f"Reddit Records:  {len(df_reddit)}")
    print(f"Files saved to:  {output_dir}")
    print("----------------")

if __name__ == "__main__":
    save_mock_data()