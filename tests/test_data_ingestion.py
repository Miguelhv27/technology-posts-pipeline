import pytest
import pandas as pd
from src.data_ingestion import DataIngestion
import os

class TestDataIngestion:
    
    def test_ingestion_initialization(self, sample_config):
        """Test that DataIngestion initializes correctly"""
        ingestion = DataIngestion(sample_config)
        
        assert ingestion.config == sample_config
        assert ingestion.raw_data_dir == sample_config["paths"]["raw_data_dir"]
    
    def test_validate_api_credentials(self, sample_config):
        """Test API credential validation"""
        ingestion = DataIngestion(sample_config)
        
        result = ingestion.validate_api_credentials()
        assert isinstance(result, bool)
    
    def test_ingestion_disabled(self, sample_config):
        """Test ingestion when disabled in config"""
        config = sample_config.copy()
        config["ingestion"]["enabled"] = False
        
        ingestion = DataIngestion(config)
        result = ingestion.run()
        
        assert result == {}