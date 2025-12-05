import pytest
import pandas as pd
import os
import sys
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def test_dirs():
    """Crea directorios temporales para los tests y los limpia al final"""
    base_dir = "tests/test_data"
    dirs = [
        os.path.join(base_dir, "raw"),
        os.path.join(base_dir, "processed"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "outputs")
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    yield dirs
    

@pytest.fixture
def sample_config():
    """
    Configuración alineada con pipeline_config.yaml v2.0.
    Simula el entorno de producción pero apuntando a carpetas de prueba.
    """
    return {
        "pipeline": {
            "name": "test_pipeline",
            "versioning": True,
            "create_timestamped_folders": False
        },
        "paths": {
            "raw_data_dir": "tests/test_data/raw",
            "processed_data_dir": "tests/test_data/processed",
            "logs_dir": "tests/test_data/logs",
            "outputs_dir": "tests/test_data/outputs"
        },
        "data_sources": {
            "twitter": {
                "enabled": True,
                "mock_data_path": "tests/test_data/raw/twitter_mock.csv"
            },
            "reddit": {
                "enabled": True,
                "kaggle_dataset": "dummy/dataset"
            }
        },
        "ingestion": {
            "enabled": True
        },
        "validation": {
            "enabled": True,
            "quality_thresholds": {
                "max_null_percentage": 0.5
            },
            "required_columns": {
                "twitter": ["id", "created_at", "text", "user_followers_count"],
                "reddit": ["post_id", "datetime", "score", "tag"] 
            },
            "check_null_columns": ["text", "score"],
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
                "sentiment": True,
                "engagement_calculation": True
            }
        },
        "nlp": {
            "enabled": True,
            "sentiment": {"use_vader": True},
            "topic_modeling": {
                "enabled": True,
                "method": "lda",
                "num_topics": 2,
                "max_df": 1.0,
                "min_df": 1
            }
        },
        "network_analysis": {
            "enabled": True,
            "influencer_threshold": 0.5,
            "metrics": ["degree_centrality"]
        },
        "analysis": {
            "enabled": True,
            "correlations": {
                "volume_vs_engagement": True,
                "influencer_activity_vs_virality": True
            },
            "statistical_tests": {
                "anova": True,
                "ttest": True
            }
        },
        "technology_posts": {
            "domains": ["AI", "Python", "Data"]
        },
        "export": {
            "format": "parquet",
            "compression": "snappy"
        },
        "alerts": {
            "sentiment_drop": {"enabled": False}
        }
    }

@pytest.fixture
def sample_twitter_data():
    """Datos Mock de Twitter con esquema correcto"""
    return pd.DataFrame({
        'id': ['tw1', 'tw2', 'tw3'],
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'text': ['AI is amazing!', 'Machine learning rocks', 'Data science future'],
        'user_id': ['user1', 'user2', 'user3'],
        'user_followers_count': [1000, 5000, 200], 
        'retweet_count': [10, 50, 2],
        'favorite_count': [20, 100, 5],
        'data_source': ['twitter', 'twitter', 'twitter']
    })

@pytest.fixture
def sample_reddit_data():
    """
    Datos Mock de Reddit imitando la estructura REAL de Kaggle.
    Usamos 'tag' en lugar de 'subreddit' y 'post_id' en lugar de 'id'.
    """
    return pd.DataFrame({
        'post_id': ['rd1', 'rd2', 'rd3'], 
        'datetime': ['2023-01-01', '2023-01-02', '2023-01-03'], 
        'title': ['AI Discussion', 'ML Tutorial', 'Data Analysis'],
        'text': ['Great post about AI', 'Learn machine learning', 'Data science insights'],
        'score': [100, 50, 75],
        'comments': [10, 5, 8],
        'tag': ['artificial', 'MachineLearning', 'datascience'], 
        'data_source': ['reddit', 'reddit', 'reddit']
    })

@pytest.fixture
def sample_processed_data(sample_twitter_data):
    """
    Datos ya transformados para probar Analysis.py sin depender de pasos previos.
    """
    df = sample_twitter_data.copy()
    df['text_cleaned'] = ['ai amazing', 'machine learning rocks', 'data science future']
    df['sentiment_compound'] = [0.8, 0.5, 0.2]
    df['sentiment_label'] = ['positive', 'positive', 'neutral']
    df['word_count'] = [2, 3, 3]
    df['total_engagement'] = [30, 150, 7]
    df['reddit_engagement'] = [0, 0, 0] 
    return df