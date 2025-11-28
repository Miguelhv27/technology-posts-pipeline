import pytest
import pandas as pd
from src.data_transformation import DataTransformation
import numpy as np

class TestDataTransformation:
    
    def test_transformation_initialization(self, sample_config):
        """Test that DataTransformation initializes correctly"""
        transformation = DataTransformation(sample_config)
        
        assert transformation.config == sample_config
        assert transformation.transformation_config == sample_config["transformation"]
    
    def test_clean_text(self, sample_config):
        """Test text cleaning functionality"""
        transformation = DataTransformation(sample_config)
        
        text = "Hello WORLD! Check this: http://example.com"
        cleaned = transformation.clean_text(text)
        
        assert "http" not in cleaned
        assert "world" in cleaned.lower()
    
    def test_extract_features(self, sample_config, sample_twitter_data):
        """Test feature extraction"""
        transformation = DataTransformation(sample_config)
        
        sample_twitter_data['text_cleaned'] = sample_twitter_data['text'].str.lower()
        
        result = transformation.extract_features(sample_twitter_data)
        
        assert 'word_count' in result.columns
        assert 'total_engagement' in result.columns
    
    def test_calculate_sentiment(self, sample_config, sample_twitter_data):
        """Test sentiment calculation"""
        transformation = DataTransformation(sample_config)
        
        sample_twitter_data['text_cleaned'] = sample_twitter_data['text'].str.lower()
        
        result = transformation.calculate_sentiment(sample_twitter_data)
        
        sentiment_cols = [col for col in result.columns if 'sentiment' in col]
        assert len(sentiment_cols) > 0
    
    def test_handle_missing_values(self, sample_config):
        """Test missing value handling"""
        transformation = DataTransformation(sample_config)
        
        data_with_missing = pd.DataFrame({
            'text': ['hello', None, 'world'],
            'score': [1, 2, None],
            'category': ['A', 'B', None]
        })
        
        result = transformation.handle_missing_values(data_with_missing)
        
        assert result['text'].isna().sum() == 0
        assert result['score'].isna().sum() == 0