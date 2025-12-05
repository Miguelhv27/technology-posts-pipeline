import pytest
import pandas as pd
import numpy as np
from src.data_transformation import DataTransformation

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
        assert "world" in cleaned 
    
    def test_extract_features(self, sample_config, sample_twitter_data):
        """Test feature extraction (cleaning, engagement, zero-imputation)"""
        transformation = DataTransformation(sample_config)
        
        sample_twitter_data.loc[0, 'user_followers_count'] = np.nan
        
        result = transformation.extract_features(sample_twitter_data)
        
        assert 'text_cleaned' in result.columns
        assert 'word_count' in result.columns
        assert 'total_engagement' in result.columns
        assert 'engagement_rate' in result.columns
        assert "ai is amazing" in result.iloc[0]['text_cleaned']
        assert result.loc[0, 'user_followers_count'] == 0
    
    def test_calculate_sentiment(self, sample_config, sample_twitter_data):
        """Test sentiment calculation"""
        transformation = DataTransformation(sample_config)

        sample_twitter_data['text_cleaned'] = sample_twitter_data['text'].apply(transformation.clean_text)
        
        result = transformation.calculate_sentiment(sample_twitter_data)
        
        assert 'sentiment_compound' in result.columns
        assert 'sentiment_label' in result.columns
        assert 'sentiment_polarity' in result.columns 
        
        assert result.iloc[0]['sentiment_label'] == 'positive'
    
    def test_handle_missing_values(self, sample_config):
        """Test missing value handling for critical pipelines"""
        transformation = DataTransformation(sample_config)
        
        data_with_missing = pd.DataFrame({
            'text_col': ['hello', None, 'world'], 
            'total_engagement': [10, 20, None],   
            'word_count': [5, None, 10],          
            'other_num': [1, 2, None]             
        })
        
        result = transformation.handle_missing_values(data_with_missing)
        
        assert result['text_col'].isna().sum() == 0
        assert result['text_col'].iloc[1] == 'unknown'
        
        assert result['total_engagement'].isna().sum() == 0
        assert result['total_engagement'].iloc[2] == 0
        
        assert result['word_count'].isna().sum() == 0
        assert result['word_count'].iloc[1] == 0

    def test_transformation_disabled(self, sample_config):
        """Test behavior when disabled"""
        config = sample_config.copy()
        config["transformation"]["enabled"] = False
        
        transformation = DataTransformation(config)
        result = transformation.run({}) 
        
        assert result is None