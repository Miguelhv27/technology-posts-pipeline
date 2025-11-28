import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

def create_twitter_mock_data():
    """Create mock Twitter data for testing """
    dates = [datetime.now() - timedelta(days=x) for x in range(200)]  
    
    twitter_data = []
    user_counter = 1000 
    
    tweets_per_day = 30
    
    for i, date in enumerate(dates):
        for j in range(tweets_per_day): 
            tweet_id = f"tw_{date.strftime('%Y%m%d')}_{j}"
            
            base_engagement = np.random.randint(5, 500)  
            retweet_count = np.random.randint(0, base_engagement // 4)
            favorite_count = np.random.randint(0, base_engagement // 2)
            
            tweet = {
                'id': tweet_id,
                'created_at': date.isoformat(),
                'text': f"Breaking news in {np.random.choice(['AI', 'Machine Learning', 'Data Science', 'Cloud Computing', 'Big Data', 'IoT', 'Blockchain', 'Neural Networks', 'Deep Learning', 'Computer Vision', 'Natural Language Processing', 'Robotics'])}! {np.random.choice(['Revolutionary breakthrough', 'Industry disruption', 'Cutting-edge innovation', 'Game-changing development'])} in this field. #Tech #{np.random.choice(['AI', 'ML', 'DataScience', 'Cloud', 'Innovation'])}",
                'user_id': f"user_{user_counter}",  
                'user_screen_name': f"tech_enthusiast_{user_counter}",
                'user_followers_count': np.random.randint(100, 50000), 
                'retweet_count': retweet_count,
                'favorite_count': favorite_count,
                'reply_count': np.random.randint(0, 50), 
                'lang': 'en',
                'source': 'twitter',
                'query': np.random.choice(['AI technology', 'machine learning', 'data science', 'cloud computing'])
            }
            twitter_data.append(tweet)
            user_counter += 1  
    
    df_twitter = pd.DataFrame(twitter_data)
    total_twitter = len(df_twitter)
    print(f"Generated {total_twitter} Twitter records")
    
    if total_twitter < 6000:
        additional_needed = 6000 - total_twitter
        print(f"Adding {additional_needed} additional Twitter records...")
        additional_data = []
        
        for i in range(additional_needed):
            base_engagement = np.random.randint(5, 300)
            
            additional_tweet = {
                'id': f"tw_extra_{i}",
                'created_at': datetime.now().isoformat(),
                'text': f"Additional tech content about {np.random.choice(['AI', 'ML', 'Data Science'])} for dataset balance.",
                'user_id': f"user_extra_{i}",
                'user_screen_name': f"extra_user_{i}",
                'user_followers_count': np.random.randint(100, 10000),
                'retweet_count': np.random.randint(0, base_engagement // 4),
                'favorite_count': np.random.randint(0, base_engagement // 2),
                'reply_count': np.random.randint(0, 20),
                'lang': 'en',
                'source': 'twitter',
                'query': 'additional'
            }
            additional_data.append(additional_tweet)
        
        df_additional = pd.DataFrame(additional_data)
        df_twitter = pd.concat([df_twitter, df_additional], ignore_index=True)
        print(f"Final Twitter dataset: {len(df_twitter)} records")
    
    df_twitter['twitter_engagement'] = df_twitter['retweet_count'] + df_twitter['favorite_count']
    avg_engagement = df_twitter['twitter_engagement'].mean()
    print(f"Twitter average engagement: {avg_engagement:.2f}")
    
    return df_twitter

def create_reddit_mock_data():
    """Create mock Reddit data for testing - MATCH TWITTER VOLUME"""
    start_date = datetime(2023, 1, 1) 
    dates = [start_date + timedelta(days=x) for x in range(180)]  
    subreddits = ['artificial', 'MachineLearning', 'datascience', 'technology', 'programming', 'computerscience', 'algorithms', 'deeplearning', 'computervision', 'nlp']
    
    reddit_data = []
    author_counter = 10000

    posts_per_subreddit_per_day = 4  
    
    for i, date in enumerate(dates):
        for subreddit in subreddits:
            for post_num in range(posts_per_subreddit_per_day):  
                post_id = f"rd_{date.strftime('%Y%m%d')}_{subreddit}_{post_num}"
                
                post = {
                    'id': post_id,
                    'created_utc': int(date.timestamp()),  
                    'title': f"Discussion about {np.random.choice(['AI', 'ML', 'Data Engineering', 'Cloud', 'Neural Networks', 'Deep Learning', 'Natural Language Processing', 'Computer Vision', 'Big Data', 'IoT'])} in {subreddit}",
                    'selftext': f"This is a comprehensive discussion about {np.random.choice(['artificial intelligence', 'machine learning algorithms', 'data engineering', 'cloud computing', 'neural networks', 'deep learning models'])}. The community is actively sharing insights about recent developments and future trends in this exciting field of technology. Many experts believe this will transform the industry in the coming years.",
                    'score': np.random.randint(10, 2500),  
                    'num_comments': np.random.randint(5, 300),  
                    'upvote_ratio': np.random.uniform(0.7, 0.95),
                    'subreddit': subreddit,
                    'author': f"user_{author_counter}",  
                    'source': 'reddit'
                }
                reddit_data.append(post)
                author_counter += 1
    
    df_reddit = pd.DataFrame(reddit_data)
    print(f"Generated {len(df_reddit)} Reddit records")
    return df_reddit

def create_commercial_mock_data():
    """Create mock commercial data for correlation analysis"""
    dates = [datetime.now() - timedelta(days=x) for x in range(90)]
    
    commercial_data = []
    for date in dates:
        commercial_data.append({
            'date': date.date(),
            'amazon_sales_velocity': np.random.randint(1000, 10000),
            'sephora_review_volume': np.random.randint(500, 5000),
            'restaurant_daily_sales': np.random.randint(2000, 15000),
            'steam_game_purchases': np.random.randint(100, 2000),
            'supplement_sales': np.random.randint(300, 3000)
        })
    
    return pd.DataFrame(commercial_data)

def save_mock_data():
    """Save mock data to raw data directory"""
    print(" Creating balanced mock datasets (6,000+ records each)...")
    
    print("Creating enhanced Twitter data...")
    df_twitter = create_twitter_mock_data()
    df_twitter.to_csv('data/raw/twitter_mock_data.csv', index=False)
    print(f" Saved {len(df_twitter)} Twitter records")
    
    print("Creating enhanced Reddit data...")
    df_reddit = create_reddit_mock_data()
    df_reddit.to_csv('data/raw/reddit_mock_data.csv', index=False)
    print(f" Saved {len(df_reddit)} Reddit records")
    
    print("Creating commercial mock data...")
    df_commercial = create_commercial_mock_data()
    df_commercial.to_csv('data/raw/commercial_data.csv', index=False)
    print(f" Saved {len(df_commercial)} commercial records")
    
    twitter_count = len(df_twitter)
    reddit_count = len(df_reddit)
    print(f"\n DATASET BALANCE:")
    print(f"   Twitter: {twitter_count:,} records")
    print(f"   Reddit:  {reddit_count:,} records")
    print(f"   Total:   {twitter_count + reddit_count:,} records")
    
    print("\n Enhanced mock data creation completed!")

if __name__ == "__main__":
    save_mock_data()