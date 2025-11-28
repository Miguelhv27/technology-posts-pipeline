import pytest
import pandas as pd
from src.analysis import AnalysisEngine
import numpy as np

class TestAnalysisEngine:
    
    def test_analysis_initialization(self, sample_config):
        """Test that AnalysisEngine initializes correctly"""
        analysis = AnalysisEngine(sample_config)
        
        assert analysis.config == sample_config
        assert analysis.analysis_config == sample_config["analysis"]
        assert analysis.technology_domains == sample_config["technology_posts"]["domains"]
    
    def test_compute_correlations(self, sample_config, sample_processed_data):
        """Test correlation computation"""
        analysis = AnalysisEngine(sample_config)
        
        sample_processed_data['reddit_engagement'] = [10, 5, 15]
        sample_processed_data['word_count'] = [10, 15, 20]
        
        correlations = analysis.compute_correlations(sample_processed_data)
        
        assert isinstance(correlations, dict)
    
    def test_perform_statistical_tests(self, sample_config, sample_processed_data):
        """Test statistical tests"""
        analysis = AnalysisEngine(sample_config)
        
        test_results = analysis.perform_statistical_tests(sample_processed_data)
        
        assert isinstance(test_results, dict)
    
    def test_generate_analysis_summary(self, sample_config):
        """Test analysis summary generation"""
        analysis = AnalysisEngine(sample_config)
        
        sample_results = {
            'correlations': {
                'test_corr': {'pearson': {'correlation': 0.8, 'p_value': 0.01}}
            },
            'statistical_tests': {
                'test_anova': {'f_statistic': 5.0, 'p_value': 0.02}
            },
            'technology_domains': {
                'domain_engagement': {'AI': {'count': 10}}
            }
        }
        
        summary = analysis.generate_analysis_summary(sample_results)
        
        assert 'total_correlations_computed' in summary
        assert 'significant_correlations' in summary
        assert 'statistical_tests_performed' in summary
    
    def test_analysis_disabled(self, sample_config, sample_processed_data):
        """Test analysis when disabled in config"""
        config = sample_config.copy()
        config["analysis"]["enabled"] = False
        
        analysis = AnalysisEngine(config)
        results = analysis.run_analysis(sample_processed_data)
        
        assert results == {}