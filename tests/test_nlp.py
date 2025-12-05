import pytest
import pandas as pd
import numpy as np
from src.nlp import NLPProcessor

class TestNLPProcessor:
    
    def test_nlp_initialization(self, sample_config):
        """Test that NLPProcessor initializes correctly"""
        nlp = NLPProcessor(sample_config)
        
        assert nlp.config == sample_config
        assert nlp.nlp_config == sample_config["nlp"]
        assert hasattr(nlp, 'lemmatizer')
        assert hasattr(nlp, 'vader_analyzer')
    
    def test_preprocess_text(self, sample_config):
        """Test text preprocessing (Cleaning + Lemmatization)"""
        nlp = NLPProcessor(sample_config)
        
        text = "This is a sample text with URL: http://example.com and Running codes!"
        processed = nlp.preprocess_text(text)

        assert "http" not in processed  
        assert "example.com" not in processed  
        
        assert processed == processed.lower()  
        
        assert "url" in processed 
        assert "sample" in processed
    
    def test_analyze_sentiment_dataframe(self, sample_config):
        """Test VADER sentiment analysis on DataFrame"""
        nlp = NLPProcessor(sample_config)
        
        df = pd.DataFrame({
            'text_cleaned': [
                "i love this amazing technology",
                "this is terrible and bad",      
                "it is a book"                    
            ]
        })
        
        df_result = nlp.analyze_sentiment(df)
        
        assert 'sentiment_compound' in df_result.columns
        assert 'sentiment_label' in df_result.columns
        
        assert df_result.iloc[0]['sentiment_label'] == 'positive'
        assert df_result.iloc[1]['sentiment_label'] == 'negative'
        assert df_result.iloc[0]['sentiment_compound'] > 0
    
    def test_extract_tech_entities(self, sample_config):
        """Test technology keyword extraction from DataFrame"""
        nlp = NLPProcessor(sample_config)
        
        sample_config['technology_posts']['domains'] = ['AI', 'Python']
        nlp = NLPProcessor(sample_config) 
        
        df = pd.DataFrame({
            'text_cleaned': [
                "learning python is great for ai", 
                "cooking pasta is fun"
            ]
        })
        
        df_result = nlp.extract_tech_entities(df)
        
        assert 'tech_keywords' in df_result.columns
        
        keywords_found = df_result.iloc[0]['tech_keywords']
        assert 'Python' in keywords_found
        assert 'AI' in keywords_found
        
        assert len(df_result.iloc[1]['tech_keywords']) == 0
    
    def test_topic_modeling_flow(self, sample_config):
        """Test LDA topic modeling execution"""
        nlp = NLPProcessor(sample_config)
        
        df = pd.DataFrame({
            'text_cleaned': ["data science machine learning"] * 15 
        })
        
        df_result, topics = nlp.perform_topic_modeling(df)
        
        assert isinstance(topics, list)
        if len(topics) > 0:
            assert isinstance(topics[0], str)
    
    def test_run_pipeline_integration(self, sample_config):
        """Test the full run() method"""
        nlp = NLPProcessor(sample_config)
        
        df = pd.DataFrame({
            'text': ["Original text about AI"],
            'title': ["Title"]
        })
        
        df_final, results = nlp.run(df)
        
        assert 'text_cleaned' in df_final.columns
        assert 'sentiment_label' in df_final.columns
        assert 'tech_keywords' in df_final.columns
        
        assert 'sentiment_distribution' in results
        assert 'topics' in results

    def test_nlp_disabled(self, sample_config):
        """Test NLP when disabled in config"""
        config = sample_config.copy()
        config["nlp"]["enabled"] = False
        
        nlp = NLPProcessor(config)
        df = pd.DataFrame({'text': ['hello world']})
        
        df_result, results = nlp.run(df)
        assert results.get('topics') == []
        assert len(df_result) == 1