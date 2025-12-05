import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from src.data_ingestion import DataIngestion

class TestDataIngestion:
    
    def test_ingestion_initialization(self, sample_config):
        """Test that DataIngestion initializes correctly"""
        ingestion = DataIngestion(sample_config)
        
        assert ingestion.config == sample_config
        assert ingestion.raw_data_dir == sample_config["paths"]["raw_data_dir"]
    
    def test_ingestion_disabled(self, sample_config):
        """Test ingestion when disabled in config"""
        config = sample_config.copy()
        config["ingestion"]["enabled"] = False
        
        ingestion = DataIngestion(config)
        result = ingestion.run()
        
        assert result == {}

    def test_fetch_twitter_mock_exists(self, sample_config):
        """Test fetching Twitter data when mock file exists"""
        # 1. Setup: Crear un mock file falso en el directorio de test
        raw_dir = sample_config["paths"]["raw_data_dir"]
        mock_path = sample_config["data_sources"]["twitter"]["mock_data_path"]
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(mock_path), exist_ok=True)
        
        # Crear archivo dummy
        df = pd.DataFrame({'id': [1], 'text': ['test tweet']})
        df.to_csv(mock_path, index=False)

        # 2. Execute
        ingestion = DataIngestion(sample_config)
        result_path = ingestion.fetch_twitter_data()

        # 3. Assert
        assert result_path == mock_path
        assert os.path.exists(result_path)

    def test_kaggle_fallback_logic(self, sample_config):
        """
        Test Crítico DataOps: 
        Simula que la API falla (o no hay auth) y verifica que el sistema
        hace 'fallback' al archivo local manual.
        """
        ingestion = DataIngestion(sample_config)
        raw_dir = ingestion.raw_data_dir
        
        # 1. Crear el archivo "Manual" (kaggle_reddit.csv)
        manual_csv = os.path.join(raw_dir, "kaggle_reddit.csv")
        df_manual = pd.DataFrame({
            'post_id': ['1', '2'], 
            'title': ['Title A', 'Title B'],
            'score': [10, 20]
        })
        df_manual.to_csv(manual_csv, index=False)

        # 2. Mockear kagglehub para que falle (Simular error 403 o Sin Internet)
        # 'side_effect' hace que la función lance una excepción cuando se llame
        with patch('src.data_ingestion.kagglehub.dataset_download', side_effect=Exception("API Connection Error")):
            
            # 3. Ejecutar descarga
            # El código debería capturar la excepción y buscar el archivo manual
            result_path = ingestion.download_kaggle_dataset()

        # 4. Assert
        assert result_path is not None
        assert "kaggle_sampled.csv" in result_path # Debe retornar el procesado, no el manual
        assert os.path.exists(result_path)
        
        # Verificar que añadió la columna de trazabilidad
        df_result = pd.read_csv(result_path)
        assert 'data_source' in df_result.columns
        assert df_result['data_source'].iloc[0] == 'reddit_kaggle'