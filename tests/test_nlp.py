import pytest
import pandas as pd
from src.nlp import NLPProcessor
import numpy as np

class TestNLPProcessor:
    
    def test_nlp_initialization(self, sample_config):
        """Test that NLPProcessor initializes correctly"""
        nlp = NLPProcessor(sample_config)
        
        assert nlp.config == sample_config
        assert nlp.nlp_config == sample_config["nlp"]
    
    def test_preprocess_text(self, sample_config):
        """Test text preprocessing"""
        nlp = NLPProcessor(sample_config)
        
        text = "This is a sample text with URL: http://example.com and SOME Capitalized WORDS!"
        processed = nlp.preprocess_text(text)

        assert "http://" not in processed  
        assert "example.com" not in processed  
        assert processed == processed.lower()  
        
        assert "://" not in processed
    
    def test_analyze_sentiment_vader(self, sample_config):
        """Test VADER sentiment analysis"""
        nlp = NLPProcessor(sample_config)
        
        positive_text = "I love this amazing product!"
        result = nlp.analyze_sentiment_vader(positive_text)
        
        assert 'compound' in result
        assert 'neg' in result  
        assert 'neu' in result   
        assert 'pos' in result  
        
        assert result['compound'] > 0
        assert result['pos'] > result['neg'] 
    
    def test_get_sentiment_label(self, sample_config):
        """Test sentiment label assignment"""
        nlp = NLPProcessor(sample_config)
        
        assert nlp.get_sentiment_label(0.8) == "positive"
        assert nlp.get_sentiment_label(-0.8) == "negative"
        assert nlp.get_sentiment_label(0.02) == "neutral"
        assert nlp.get_sentiment_label(-0.02) == "neutral"
    
    def test_extract_technology_keywords(self, sample_config):
        """Test technology keyword extraction"""
        nlp = NLPProcessor(sample_config)
        
        text = "This is about Artificial Intelligence and Machine Learning"
        keywords = nlp.extract_technology_keywords(text)
        
        assert "Artificial Intelligence" in keywords
        assert "Machine Learning" in keywords
    
    def test_nlp_disabled(self, sample_config):
        """Test NLP when disabled in config"""
        config = sample_config.copy()
        config["nlp"]["enabled"] = False
        
        nlp = NLPProcessor(config)
        assert nlp.config == config