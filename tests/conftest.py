import pytest
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_config():
    """Sample configuration for testing - UPDATED with missing sections"""
    return {
        "pipeline": {
            "name": "test_pipeline",
            "versioning": True,
            "create_timestamped_folders": True
        },
        "paths": {
            "raw_data_dir": "tests/test_data/raw",
            "processed_data_dir": "tests/test_data/processed",
            "logs_dir": "tests/test_data/logs",
            "models_dir": "tests/test_data/models",
            "outputs_dir": "tests/test_data/outputs"
        },
        "data_sources": {
            "twitter": {
                "enabled": True,
                "search_queries": ["AI", "Machine Learning"],
                "max_tweets": 100
            },
            "reddit": {
                "enabled": True,
                "subreddits": ["artificial", "MachineLearning"]
            }
        },
        "apis": {
            "twitter": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "bearer_token": "test_token"
            },
            "reddit": {
                "client_id": "test_client",
                "client_secret": "test_secret"
            }
        },
        "ingestion": {
            "enabled": True,
            "overwrite_raw": False
        },
        "validation": {
            "enabled": True,
            "required_columns": {
                "twitter": ["id", "created_at", "text", "user_id"],
                "reddit": ["id", "created_utc", "title", "score"]
            },
            "check_null_columns": ["title", "text"],
            "quality_thresholds": {
                "max_null_percentage": 0.1
            },
            "allowed_date_range": {
                "start": "2015-01-01",
                "end": "2030-01-01"
            }
        },
        "transformation": {
            "enabled": True,
            "text_cleaning": {
                "lowercase": True,
                "remove_urls": True,
                "remove_special_chars": True
            },
            "feature_engineering": {
                "word_count": True,
                "sentiment": True,
                "engagement_metrics": True,
                "time_features": True
            }
        },
        "nlp": {
            "enabled": True,
            "sentiment": {
                "use_vader": True,
                "min_confidence": 0.7
            },
            "topic_modeling": {
                "method": "lda",
                "num_topics": 5,
                "max_features": 1000,
                "min_df": 2,
                "max_df": 0.8
            },
            "entity_recognition": {
                "enabled": True,
                "entities": ["PERSON", "ORG"]
            }
        },
        "network_analysis": {
            "enabled": True,
            "platform": "twitter",
            "metrics": ["pagerank", "degree_centrality"],
            "influencer_threshold": 0.5,
            "min_connections": 2,
            "community_detection": True
        },
        "analysis": {
            "enabled": True,
            "correlation_method": "pearson",
            "min_correlation_threshold": 0.6,
            "correlations": {
                "volume_vs_engagement": True,
                "influencer_activity_vs_virality": True,
                "sentiment_vs_commercial_metrics": True
            },
            "statistical_tests": {
                "anova": True,
                "ttest": True,
                "normality_tests": True
            },
            "time_window": "7d"
        },
        "timeseries": {
            "enabled": True,
            "frequency": "D",
            "decomposition": {
                "apply": True,
                "model": "additive",
                "period": 7
            }
        },
        "observability": {
            "logging_level": "INFO",
            "metrics": {
                "track_runtime": True,
                "track_memory": True
            }
        },
        "alerts": {
            "sentiment_drop": {
                "enabled": True,
                "threshold": -0.3
            }
        },
        "export": {
            "save_parquet": True,
            "compression": "snappy"
        },

        "technology_posts": {
            "domains": [
                "Artificial Intelligence",
                "Machine Learning", 
                "Data Engineering",
                "Cloud Computing"
            ],
            "key_metrics": {
                "primary": "engagement_rate",
                "secondary": "sentiment_score"
            }
        }
    }

@pytest.fixture
def sample_twitter_data():
    """Sample Twitter data for testing"""
    return pd.DataFrame({
        'id': ['1', '2', '3'],
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'text': ['AI is amazing!', 'Machine learning rocks', 'Data science future'],
        'user_id': ['user1', 'user2', 'user3'],
        'retweet_count': [10, 5, 15],
        'favorite_count': [20, 10, 25],
        'data_source': ['twitter', 'twitter', 'twitter']
    })

@pytest.fixture
def sample_reddit_data():
    """Sample Reddit data for testing"""
    return pd.DataFrame({
        'id': ['1', '2', '3'],
        'created_utc': [1672531200, 1672617600, 1672704000],
        'title': ['AI Discussion', 'ML Tutorial', 'Data Analysis'],
        'selftext': ['Great post about AI', 'Learn machine learning', 'Data science insights'],
        'score': [100, 50, 75],
        'num_comments': [10, 5, 8],
        'subreddit': ['artificial', 'MachineLearning', 'datascience'],
        'data_source': ['reddit', 'reddit', 'reddit']
    })

@pytest.fixture
def sample_processed_data():
    """Sample processed data for testing"""
    return pd.DataFrame({
        'id': ['1', '2', '3'],
        'text_cleaned': ['ai amazing', 'machine learning rocks', 'data science future'],
        'sentiment_polarity': [0.5, 0.3, 0.7],
        'word_count': [2, 3, 3],
        'total_engagement': [30, 15, 40],
        'reddit_engagement': [10, 5, 15],
        'data_source': ['twitter', 'twitter', 'reddit'],
        'sentiment_label': ['positive', 'neutral', 'positive']
    })