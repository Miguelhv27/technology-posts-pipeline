import pytest
import os
import yaml
import pandas as pd
import sys
from unittest.mock import patch, MagicMock
from src.orchestrator import Orchestrator

class TestOrchestrator:
    
    def test_orchestrator_initialization(self, sample_config, tmp_path):
        """Test that Orchestrator initializes correctly with a real config file"""
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
        
        orchestrator = Orchestrator(str(config_path))
        
        assert orchestrator.config == sample_config
        assert orchestrator.processed_data_dir == sample_config["paths"]["processed_data_dir"]
        assert orchestrator.raw_data_dir == sample_config["paths"]["raw_data_dir"]
    
    def test_orchestrator_missing_config(self):
        """
        Test that Orchestrator calls sys.exit(1) on missing config
        (Updated logic from FileNotFoundError to SystemExit)
        """
        with pytest.raises(SystemExit) as excinfo:
            Orchestrator("non_existent_config.yaml")
        
        assert excinfo.value.code == 1
    
    def test_create_output_folders(self, sample_config, tmp_path):
        """Test folder creation logic (Renamed method)"""
        sample_config["paths"]["outputs_dir"] = str(tmp_path / "outputs")
        sample_config["paths"]["raw_data_dir"] = str(tmp_path / "raw")
        sample_config["paths"]["processed_data_dir"] = str(tmp_path / "processed")
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)
            
        orchestrator = Orchestrator(str(config_path))
        
        folder = orchestrator.create_output_folders()
        
        assert os.path.exists(folder)
        assert os.path.exists(orchestrator.raw_data_dir)
        assert os.path.exists(orchestrator.processed_data_dir)

    @patch('src.orchestrator.ingest_data')
    @patch('src.orchestrator.validate_raw_data')
    @patch('src.orchestrator.transform_data')
    @patch('src.orchestrator.run_nlp_pipeline')
    @patch('src.orchestrator.run_network_analysis')
    @patch('src.orchestrator.run_statistical_analysis')
    def test_orchestrator_full_run_mock(self, mock_stats, mock_net, mock_nlp, mock_trans, mock_val, mock_ingest, sample_config, tmp_path):
        """
        Integration Test con Mocks:
        Verifica que el orquestador llame a los pasos en el orden correcto
        sin ejecutar realmente el procesamiento pesado.
        """

        mock_ingest.return_value = {'twitter': 'path/to/twitter.csv'}
        mock_val.return_value = True
        mock_trans.return_value = pd.DataFrame({'id': [1]}) 
        mock_nlp.return_value = (pd.DataFrame({'id': [1]}), {'topics': []})
        mock_net.return_value = {'metrics': {}}
        mock_stats.return_value = {'correlations': {}}

        config_path = tmp_path / "test_config.yaml"
        sample_config["paths"]["outputs_dir"] = str(tmp_path)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(sample_config, f)

        orchestrator = Orchestrator(str(config_path))
        success = orchestrator.run()

        assert success is True
        mock_ingest.assert_called_once()
        mock_val.assert_called_once()
        mock_trans.assert_called_once()
        mock_nlp.assert_called_once()
        mock_net.assert_called_once()
        mock_stats.assert_called_once()