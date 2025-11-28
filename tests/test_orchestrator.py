import pytest
import os
import yaml
from orchestrator import Orchestrator

class TestOrchestrator:
    
    def test_orchestrator_initialization(self, sample_config, tmp_path):
        """Test that Orchestrator initializes correctly with a real config file"""
        config_path = tmp_path / "test_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        orchestrator = Orchestrator(str(config_path))
        
        assert orchestrator.config == sample_config
        assert orchestrator.processed_data_dir == sample_config["paths"]["processed_data_dir"]
        assert orchestrator.raw_data_dir == sample_config["paths"]["raw_data_dir"]
    
    def test_orchestrator_missing_config(self):
        """Test that Orchestrator handles missing config file properly"""
        with pytest.raises(FileNotFoundError):
            Orchestrator("non_existent_config.yaml")
    
    def test_create_versioned_output_folder(self, sample_config, tmp_path):
        """Test versioned folder creation"""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f)
        
        orchestrator = Orchestrator(str(config_path))
        
        folder = orchestrator.create_versioned_output_folder()
        
        assert os.path.exists(folder)
        assert "processed" in folder
    
    def test_orchestrator_with_default_config(self):
        """Test orchestrator with default config path"""
        try:
            orchestrator = Orchestrator()  
            assert orchestrator.config is not None
        except FileNotFoundError:
            pass