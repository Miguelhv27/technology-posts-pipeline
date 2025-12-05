import logging
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class NLPProcessor:
    """
    DataOps NLP Engine.
    Responsibilities:
    1. Text Cleaning & Normalization
    2. Sentiment Analysis (VADER) - Optimized for Social Media
    3. Topic Modeling (LDA) - Discovery of latent themes
    """

    def __init__(self, config):
        self.config = config
        self.nlp_config = config["nlp"]
        self.technology_domains = config["technology_posts"]["domains"]

        self._initialize_resources()

    def _initialize_resources(self):
        """Load NLTK resources and VADER analyzer securely"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logging.info("Downloading NLTK resources...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """
        Robust text cleaning pipeline:
        Lowercasing -> URL removal -> Special Char removal -> Lemmatization
        """
        if pd.isna(text) or str(text).strip() == "":
            return ""

        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S*\.com\S*|\S*\.org\S*', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        try:
            tokens = word_tokenize(text)
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and len(token) > 2
            ]
            return " ".join(tokens)
        except Exception:
            return text

    def analyze_sentiment(self, df):
        """
        Applies VADER Sentiment Analysis.
        Adds columns: 'sentiment_compound', 'sentiment_label'
        """
        logging.info("Running VADER Sentiment Analysis...")
        
        if 'text_cleaned' not in df.columns:
            logging.warning("Text column not found via NLP preprocessing. Skipping sentiment.")
            return df

        
        def get_vader_score(text):
            if not text: return 0.0
            return self.vader_analyzer.polarity_scores(text)['compound']

        df['sentiment_compound'] = df['text_cleaned'].apply(get_vader_score)
        
        conditions = [
            (df['sentiment_compound'] >= 0.05),
            (df['sentiment_compound'] <= -0.05)
        ]
        choices = ['positive', 'negative']
        df['sentiment_label'] = np.select(conditions, choices, default='neutral')
        
        df['sentiment_polarity'] = df['sentiment_compound']
        
        return df

    def perform_topic_modeling(self, df):
        """
        Extracts latent topics using LDA (Latent Dirichlet Allocation).
        Adds column: 'dominant_topic_keywords'
        """
        if not self.nlp_config.get("topic_modeling", {}).get("enabled", False):
            return df, []

        logging.info("Running LDA Topic Modeling...")
        
        clean_docs = df['text_cleaned'].dropna()
        clean_docs = clean_docs[clean_docs.str.len() > 10] 
        
        if len(clean_docs) < 10:
            logging.warning("Not enough data for Topic Modeling.")
            return df, []

        try:
            vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
            dtm = vectorizer.fit_transform(clean_docs)

            num_topics = self.nlp_config["topic_modeling"].get("num_topics", 3)
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(dtm)
            
            feature_names = vectorizer.get_feature_names_out()
            topics_summary = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_features_ind = topic.argsort()[:-6:-1] 
                keywords = [feature_names[i] for i in top_features_ind]
                topics_summary.append(f"Topic {topic_idx}: {', '.join(keywords)}")
            
            logging.info(f"Topics discovered: {topics_summary}")
            return df, topics_summary

        except Exception as e:
            logging.error(f"Topic Modeling failed: {e}")
            return df, []

    def extract_tech_entities(self, df):
        """Simple keyword matching for Technology Domains"""
        def find_domains(text):
            found = [d for d in self.technology_domains if d.lower() in str(text).lower()]
            return found if found else []

        df['tech_keywords'] = df['text_cleaned'].apply(find_domains)
        return df

    def run(self, df):
        """Main execution pipeline"""
        if df is None or df.empty:
            return df, {}

        if 'text' in df.columns:
            df['text_cleaned'] = df['text'].apply(self.preprocess_text)
        elif 'title' in df.columns:
            df['text_cleaned'] = df['title'].apply(self.preprocess_text)
        else:
            logging.error("No text column found for NLP.")
            return df, {}

        df = self.analyze_sentiment(df)
        
        df, topics = self.perform_topic_modeling(df)
        
        df = self.extract_tech_entities(df)
        
        results = {
            "topics": topics,
            "sentiment_distribution": df['sentiment_label'].value_counts().to_dict()
        }
        
        return df, results

def run_nlp_pipeline(df, config):
    processor = NLPProcessor(config)
    return processor.run(df)