import os
import logging
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class NLPProcessor:
    """
    Advanced NLP pipeline for sentiment analysis and topic modeling
    for Technology Posts analysis.
    """

    def __init__(self, config):
        self.config = config
        self.nlp_config = config["nlp"]
        self.technology_domains = config["technology_posts"]["domains"]
        
        self.setup_nlp_components()
        
        self.download_nltk_resources()

    def setup_nlp_components(self):
        """Initialize NLP models and components"""

        if self.nlp_config["sentiment"]["use_vader"]:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        if self.nlp_config["topic_modeling"]["method"] == "lda":
            self.vectorizer = CountVectorizer(
                max_features=self.nlp_config["topic_modeling"]["max_features"],
                min_df=self.nlp_config["topic_modeling"]["min_df"],
                max_df=self.nlp_config["topic_modeling"]["max_df"],
                stop_words='english'
            )
            
            self.lda_model = LatentDirichletAllocation(
                n_components=self.nlp_config["topic_modeling"]["num_topics"],
                random_state=42
            )
        
        self.lemmatizer = WordNetLemmatizer()

    def download_nltk_resources(self):
        """Download required NLTK datasets"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def preprocess_text(self, text):
        """Advanced text preprocessing for NLP tasks"""
        if pd.isna(text) or len(str(text).strip()) == 0:
            return ""
        
        text = str(text)
        
        tokens = word_tokenize(text.lower())
        
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        processed_text = " ".join(tokens)
        
        processed_text = re.sub(r'http\S+|www\S+|https\S+', '', processed_text, flags=re.MULTILINE)
        processed_text = re.sub(r'\S*\.com\S*|\S*\.org\S*|\S*\.net\S*', '', processed_text)
        
        processed_text = re.sub(r'[^a-zA-Z\s]', '', processed_text)
        
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()
        
        return processed_text

    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {"compound": 0.0, "neg": 0.0, "neu": 1.0, "pos": 0.0}
        
        return self.vader_analyzer.polarity_scores(str(text))

    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        if pd.isna(text) or len(str(text).strip()) == 0:
            return {"polarity": 0.0, "subjectivity": 0.0}
        
        analysis = TextBlob(str(text))
        return {
            "polarity": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity
        }

    def get_sentiment_label(self, score):
        """Sentiment classification """
        if abs(score) < 0.01 or score == 0.0:
            return "neutral"
        elif score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    def perform_sentiment_analysis(self, df):
        """Perform comprehensive sentiment analysis - FIXED VERSION"""
        logging.info("Realizando análisis de sentimiento")
        
        text_columns = [col for col in df.columns if col.endswith('_cleaned')]
        
        for text_col in text_columns:
            base_col = text_col.replace('_cleaned', '')
            
            if self.nlp_config["sentiment"]["use_vader"]:
                sentiment_scores = df[text_col].apply(self.analyze_sentiment_vader)
                
                df[f'{base_col}_vader_compound'] = sentiment_scores.apply(lambda x: x['compound'])
                df[f'{base_col}_vader_positive'] = sentiment_scores.apply(lambda x: x['pos'])  
                df[f'{base_col}_vader_negative'] = sentiment_scores.apply(lambda x: x['neg'])  
                df[f'{base_col}_vader_neutral'] = sentiment_scores.apply(lambda x: x['neu'])   
                df[f'{base_col}_sentiment_label'] = df[f'{base_col}_vader_compound'].apply(self.get_sentiment_label)
            
            if not self.nlp_config["sentiment"]["use_vader"]:
                blob_scores = df[text_col].apply(self.analyze_sentiment_textblob)
                df[f'{base_col}_blob_polarity'] = blob_scores.apply(lambda x: x['polarity'])
                df[f'{base_col}_blob_subjectivity'] = blob_scores.apply(lambda x: x['subjectivity'])
                df[f'{base_col}_sentiment_label'] = df[f'{base_col}_blob_polarity'].apply(self.get_sentiment_label)
        
        return df

    def extract_technology_keywords(self, text):
        """Extract technology-related keywords"""
        if pd.isna(text):
            return []
        
        text_lower = str(text).lower()
        found_keywords = []
        
        for domain in self.technology_domains:
            domain_lower = domain.lower()
            if domain_lower in text_lower:
                found_keywords.append(domain)
        
        return found_keywords

    def perform_topic_modeling(self, df):
        """Perform LDA topic modeling on text data"""
        if not self.nlp_config["topic_modeling"]["method"] == "lda":
            return df, None
        
        logging.info("Realizando modelado de temas (LDA)")
        
        text_columns = [col for col in df.columns if col.endswith('_cleaned')]
        if not text_columns:
            logging.warning("No hay columnas de texto para modelado de temas")
            return df, None
        
        main_text_col = text_columns[0]
        
        processed_texts = df[main_text_col].apply(self.preprocess_text)
        
        non_empty_texts = processed_texts[processed_texts.str.len() > 0]
        if len(non_empty_texts) == 0:
            logging.warning("No hay textos válidos para modelado de temas")
            return df, None
        
        try:
            tf_features = self.vectorizer.fit_transform(non_empty_texts)
            
            lda_output = self.lda_model.fit_transform(tf_features)
            
            topic_assignments = np.argmax(lda_output, axis=1)
            
            topic_mapping = pd.Series(index=non_empty_texts.index, data=topic_assignments)
            df['dominant_topic'] = topic_mapping
            
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_features_idx = topic.argsort()[:-10 - 1:-1]
                top_features = [feature_names[i] for i in top_features_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'top_keywords': top_features
                })
            
            logging.info(f"LDA completado con {len(topics)} temas")
            return df, topics
            
        except Exception as e:
            logging.error(f"Error en modelado de temas: {e}")
            return df, None

    def extract_technology_entities(self, df):
        """Extract technology-related entities and keywords"""
        logging.info("Extrayendo entidades tecnológicas")
        
        text_columns = [col for col in df.columns if col.endswith('_cleaned')]
        
        for text_col in text_columns:
            base_col = text_col.replace('_cleaned', '')
            df[f'{base_col}_tech_keywords'] = df[text_col].apply(self.extract_technology_keywords)
            df[f'{base_col}_tech_keyword_count'] = df[f'{base_col}_tech_keywords'].apply(len)
        
        return df

    def calculate_engagement_sentiment_correlation(self, df):
        """Calculate correlation between sentiment and engagement"""
        logging.info("Calculando correlaciones sentimiento-engagement")
        
        sentiment_cols = [col for col in df.columns if 'vader_compound' in col or 'blob_polarity' in col]
        engagement_cols = [col for col in df.columns if 'engagement' in col.lower() or 'score' in col or 'retweet' in col]
        
        correlations = {}
        
        for sent_col in sentiment_cols:
            for eng_col in engagement_cols:
                if sent_col in df.columns and eng_col in df.columns:
                    corr = df[sent_col].corr(df[eng_col])
                    correlations[f"{sent_col}_{eng_col}"] = corr
        
        return correlations

    def run_nlp_pipeline(self, df):
        """Execute complete NLP pipeline"""
        logging.info("Iniciando pipeline NLP avanzado")
        
        if not self.nlp_config["enabled"]:
            logging.info("NLP desactivado en configuración")
            return df, {}
        
        try:
            df = self.perform_sentiment_analysis(df)
            
            if self.nlp_config["entity_recognition"]["enabled"]:
                df = self.extract_technology_entities(df)
            
            df, topics = self.perform_topic_modeling(df)
            
            correlations = self.calculate_engagement_sentiment_correlation(df)
            
            nlp_results = {
                'topics': topics,
                'correlations': correlations,
                'sentiment_stats': {
                    'mean_sentiment': df[[col for col in df.columns if 'vader_compound' in col or 'blob_polarity' in col]].mean().to_dict(),
                    'sentiment_distribution': df[[col for col in df.columns if 'sentiment_label' in col]].value_counts().to_dict()
                }
            }
            
            logging.info("Pipeline NLP completado exitosamente")
            return df, nlp_results
            
        except Exception as e:
            logging.error(f"Error en pipeline NLP: {e}")
            raise

    def run(self, df):
        return self.run_nlp_pipeline(df)

def run_nlp_pipeline(df, config):
    """Main NLP pipeline function"""
    nlp_processor = NLPProcessor(config)
    return nlp_processor.run(df)
