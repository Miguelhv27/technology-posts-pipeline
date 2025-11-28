import pytest
import pandas as pd
from src.data_validation import DataValidation
import os

class TestDataValidation:
    
    def test_validation_initialization(self, sample_config):
        """Test that DataValidation initializes correctly"""
        validation = DataValidation(sample_config)
        
        assert validation.config == sample_config
        assert validation.validation_config == sample_config["validation"]
    
    def test_validate_columns_twitter(self, sample_config, sample_twitter_data):
        """Test column validation for Twitter data"""
        validation = DataValidation(sample_config)
        
        valid, missing_cols = validation.validate_columns(
            sample_twitter_data, "test_twitter.csv", "twitter"
        )
        
        assert valid == True
        assert missing_cols == []
    
    def test_validate_missing_values(self, sample_config, sample_twitter_data):
        """Test missing value validation"""
        validation = DataValidation(sample_config)
        
        valid, missing_stats = validation.validate_missing_values(
            sample_twitter_data, "test.csv"
        )
        
        assert valid == True
    
    def test_detect_source_type(self, sample_config):
        """Test source type detection"""
        validation = DataValidation(sample_config)
        
        twitter_data = pd.DataFrame({'retweet_count': [1, 2], 'favorite_count': [3, 4]})
        source_type = validation.detect_source_type("twitter_data.csv", twitter_data)
        assert source_type == "twitter"
    
    def test_validation_disabled(self, sample_config):
        """Test validation when disabled in config"""
        config = sample_config.copy()
        config["validation"]["enabled"] = False
        
        validation = DataValidation(config)
        result = validation.run()
        
        assert result["overall_status"] == "SKIPPED"