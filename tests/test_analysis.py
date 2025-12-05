import pytest
import pandas as pd
import numpy as np
from src.analysis import AnalysisEngine, run_statistical_analysis

class TestAnalysisEngine:
    
    def test_analysis_initialization(self, sample_config):
        """Test that AnalysisEngine initializes correctly"""
        analysis = AnalysisEngine(sample_config)
        
        assert analysis.config == sample_config
        assert analysis.analysis_config == sample_config["analysis"]
        assert analysis.technology_domains == sample_config["technology_posts"]["domains"]
    
    def test_statistical_intelligence(self, sample_config):
        """
        Prueba CRÍTICA: Verificar que el motor elige el test adecuado.
        """
        analysis = AnalysisEngine(sample_config)
        
        # Caso 1: Datos Normales (Debería elegir Pearson)
        # Generamos una distribución normal perfecta
        np.random.seed(42)
        x_norm = np.random.normal(0, 1, 100)
        y_norm = x_norm * 2 + np.random.normal(0, 0.1, 100)
        
        res_pearson = analysis._get_optimal_correlation(x_norm, y_norm)
        assert res_pearson['method_used'] == 'pearson'
        assert res_pearson['correlation'] > 0.9
        
        # Caso 2: Datos NO Normales (Debería elegir Spearman)
        # Generamos datos exponenciales/sesgados
        x_skew = np.random.exponential(1, 100)
        y_skew = x_skew ** 2  # Relación monótona pero no lineal
        
        res_spearman = analysis._get_optimal_correlation(x_skew, y_skew)
        assert res_spearman['method_used'] == 'spearman'
        assert res_spearman['correlation'] > 0.9

    def test_compute_correlations(self, sample_config, sample_processed_data):
        """Test correlation computation with full pipeline logic"""
        analysis = AnalysisEngine(sample_config)
        
        # Asegurar datos suficientes para test estadístico (>3 filas)
        # Extendemos el fixture que solo traía 3 filas
        df_extended = pd.concat([sample_processed_data] * 5, ignore_index=True)
        
        # Simular correlación fuerte
        df_extended['word_count'] = range(len(df_extended))
        df_extended['reddit_engagement'] = [x * 2 for x in range(len(df_extended))]
        
        correlations = analysis.compute_correlations(df_extended)
        
        assert isinstance(correlations, dict)
        assert 'volume_vs_engagement' in correlations
        assert correlations['volume_vs_engagement']['significant'] 

    def test_perform_statistical_tests(self, sample_config, sample_processed_data):
        """Test ANOVA/Kruskal logic"""
        analysis = AnalysisEngine(sample_config)
        
        # Crear grupos para comparar
        df = pd.DataFrame({
            'sentiment_label': ['positive']*10 + ['negative']*10 + ['neutral']*10,
            'total_engagement': np.concatenate([
                np.random.normal(100, 10, 10), # Positivos altos
                np.random.normal(50, 10, 10),  # Negativos bajos
                np.random.normal(20, 5, 10)    # Neutros muy bajos
            ])
        })
        
        test_results = analysis.perform_statistical_tests(df)
        
        assert 'sentiment_engagement_comparison' in test_results
        result = test_results['sentiment_engagement_comparison']
        assert result['p_value'] < 0.05 # Debería ser significativo por diseño
        # El test usado dependerá de la varianza, ambos son válidos aquí
        assert result['test_used'] in ["ANOVA (Fisher)", "Kruskal-Wallis (Heteroscedastic)"]

    def test_generate_analysis_summary(self, sample_config):
        """Test analysis summary generation with new structure"""
        analysis = AnalysisEngine(sample_config)
        
        # Simular resultados en el nuevo formato plano
        sample_results = {
            'correlations': {
                'volume_vs_engagement': {
                    'correlation': 0.8, 
                    'p_value': 0.01, 
                    'method_used': 'pearson',
                    'significant': True
                }
            },
            'statistical_tests': {
                'anova_test': {'statistic': 5.0, 'p_value': 0.02}
            },
            'technology_domains': {
                'domain_engagement': {'AI': {'count': 10}}
            }
        }
        
        summary = analysis.generate_analysis_summary(sample_results)
        
        assert summary['status'] == 'completed'
        assert summary['counts']['correlations'] == 1
        assert summary['counts']['statistical_tests'] == 1
        # Verificar que detectó el hallazgo significativo
        assert len(summary['significant_findings']) > 0
        assert "volume_vs_engagement" in summary['significant_findings'][0]
    
    def test_analysis_disabled(self, sample_config, sample_processed_data):
        """Test analysis when disabled in config"""
        config = sample_config.copy()
        config["analysis"]["enabled"] = False
        
        analysis = AnalysisEngine(config)
        results = analysis.run_analysis(sample_processed_data)
        
        assert results == {}

    def test_run_statistical_analysis_wrapper(self, sample_config, sample_processed_data):
        """Integration test for the main wrapper function"""
        # Datos mínimos
        df = sample_processed_data
        network_results = {}
        
        results = run_statistical_analysis(df, network_results, sample_config)
        
        assert 'correlations' in results
        assert 'statistical_tests' in results
        assert 'summary' in results