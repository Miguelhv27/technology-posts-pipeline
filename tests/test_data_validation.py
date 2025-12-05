import pytest
import pandas as pd
import numpy as np
from src.data_validation import DataValidation

class TestDataValidation:
    
    def test_validation_initialization(self, sample_config):
        """Test that DataValidation initializes correctly"""
        validation = DataValidation(sample_config)
        
        assert validation.config == sample_config
        assert validation.validation_config == sample_config["validation"]
    
    def test_validate_schema_twitter(self, sample_config, sample_twitter_data):
        """
        Test schema validation logic (Renamed from validate_columns).
        Checks if required columns like 'user_followers_count' are present.
        """
        validation = DataValidation(sample_config)
        
        valid, missing_cols = validation.validate_schema(
            sample_twitter_data, "test_twitter.csv", "twitter"
        )
        assert valid is True
        assert missing_cols == []

        bad_df = sample_twitter_data.drop(columns=['user_followers_count'])
        valid, missing_cols = validation.validate_schema(
            bad_df, "bad_twitter.csv", "twitter"
        )
        assert valid is False
        assert 'user_followers_count' in missing_cols
    
    def test_validate_data_quality(self, sample_config, sample_twitter_data):
        """
        Test data quality checks (Renamed from validate_missing_values).
        Verifies null checks and zero variance.
        """
        validation = DataValidation(sample_config)
        
        valid, issues = validation.validate_data_quality(
            sample_twitter_data, "clean.csv"
        )
        assert valid is True
        assert issues == {}

        dirty_df = sample_twitter_data.copy()
        dirty_df.loc[:, 'user_followers_count'] = np.nan 
        
        valid, issues = validation.validate_data_quality(
            dirty_df, "dirty.csv"
        )
        assert 'user_followers_count_nulls' in issues
    
    def test_detect_source_type(self, sample_config):
        """Test robust source detection logic"""
        validation = DataValidation(sample_config)
        
        twitter_df = pd.DataFrame({'retweet_count': [1], 'user_followers_count': [100]})
        source_type = validation.detect_source_type(twitter_df, "random_name.csv")
        assert source_type == "twitter"

        reddit_df = pd.DataFrame({'score': [1]})
        source_type = validation.detect_source_type(reddit_df, "kaggle_reddit.csv")
        assert source_type == "reddit"
    
    def test_validation_disabled(self, sample_config):
        """Test validation skipping mechanism"""
        config = sample_config.copy()
        config["validation"]["enabled"] = False
        
        validation = DataValidation(config)
        result = validation.run()
        
        assert result["overall_status"] == "SKIPPED"