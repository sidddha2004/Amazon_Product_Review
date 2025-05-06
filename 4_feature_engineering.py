# 4_feature_engineering.py - Feature extraction and engineering for Amazon Review Sentiment Analysis

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

class FeatureEngineer:
    """Class for extracting and engineering features from review text"""
    
    def __init__(self, max_features=5000):
        """Initialize feature engineering components
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
        """
        self.max_features = max_features
        # For small datasets, use different min_df and max_df settings
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=1,  # Minimum document frequency - just 1 for small datasets
            max_df=1.0,  # Maximum document frequency - allow all for small datasets
            ngram_range=(1, 2)  # Use both unigrams and bigrams
        )
        self.fitted = False
        
    def fit_tfidf(self, texts):
        """Fit TF-IDF vectorizer on training texts
        
        Args:
            texts (array-like): List of text documents
            
        Returns:
            self: For method chaining
        """
        print(f"Fitting TF-IDF vectorizer with max_features={self.max_features}...")
        self.tfidf_vectorizer.fit(texts)
        self.fitted = True
        print(f"Fitted TF-IDF vectorizer with {len(self.tfidf_vectorizer.get_feature_names_out())} features")
        return self
    
    def transform_tfidf(self, texts):
        """Transform texts to TF-IDF features
        
        Args:
            texts (array-like): List of text documents
            
        Returns:
            sparse matrix: TF-IDF features
        """
        if not self.fitted:
            raise ValueError("TF-IDF vectorizer not fitted yet. Call fit_tfidf first.")
        return self.tfidf_vectorizer.transform(texts)
    
    def get_top_features(self, n=20):
        """Get top TF-IDF features
        
        Args:
            n (int): Number of top features to return
            
        Returns:
            list: Top feature names
        """
        if not self.fitted:
            raise ValueError("TF-IDF vectorizer not fitted yet. Call fit_tfidf first.")
            
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        return feature_names[:n]
    
    def extract_sentiment_lexicon_features(self, df, column='cleaned_text'):
        """Extract features based on sentiment lexicons
        
        Args:
            df (pd.DataFrame): DataFrame with text column
            column (str): Column name containing text
            
        Returns:
            pd.DataFrame: DataFrame with added lexicon features
        """
        print("Extracting sentiment lexicon features...")
        
        # Simple lexicons (expand these with proper sentiment lexicons)
        positive_words = [
            'great', 'good', 'excellent', 'amazing', 'love', 'best', 'perfect',
            'awesome', 'fantastic', 'wonderful', 'happy', 'pleased', 'delighted',
            'outstanding', 'superior', 'terrific', 'super', 'impressive', 'exceptional'
        ]
        
        negative_words = [
            'bad', 'terrible', 'poor', 'worst', 'hate', 'awful', 'horrible',
            'disappointing', 'useless', 'waste', 'broken', 'defective', 'faulty',
            'pathetic', 'mediocre', 'inferior', 'sucks', 'lousy', 'subpar'
        ]
        
        # Count positive and negative words
        df['positive_word_count'] = df[column].apply(
            lambda x: sum(1 for word in positive_words if word in x.lower().split())
        )
        
        df['negative_word_count'] = df[column].apply(
            lambda x: sum(1 for word in negative_words if word in x.lower().split())
        )
        
        # Sentiment ratio
        df['sentiment_ratio'] = df.apply(
            lambda x: (x['positive_word_count'] + 1) / (x['negative_word_count'] + 1),
            axis=1
        )
        
        print("Sentiment lexicon features extracted.")
        return df
    
    def extract_syntactic_features(self, df, column='cleaned_text'):
        """Extract syntactic features from text
        
        Args:
            df (pd.DataFrame): DataFrame with text column
            column (str): Column name containing text
            
        Returns:
            pd.DataFrame: DataFrame with added syntactic features
        """
        print("Extracting syntactic features...")
        
        # Text length features
        df['text_length'] = df[column].apply(len)
        df['word_count'] = df[column].apply(lambda x: len(x.split()))
        df['avg_word_length'] = df[column].apply(
            lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
        )
        
        # Punctuation features (based on original text if available)
        orig_text_col = 'reviews.text' if 'reviews.text' in df.columns else column
        df['exclamation_count'] = df[orig_text_col].apply(
            lambda x: x.count('!') if isinstance(x, str) else 0
        )
        df['question_count'] = df[orig_text_col].apply(
            lambda x: x.count('?') if isinstance(x, str) else 0
        )
        
        print("Syntactic features extracted.")
        return df
    
    def extract_all_features(self, df, text_column='processed_text'):
        """Extract all available features from DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            text_column (str): Column containing processed text
            
        Returns:
            pd.DataFrame: DataFrame with all features added
        """
        print("Extracting all features...")
        
        # Extract lexicon features
        df = self.extract_sentiment_lexicon_features(df, text_column)
        
        # Extract syntactic features
        df = self.extract_syntactic_features(df, text_column)
        
        # Add title features if available
        if 'reviews.title' in df.columns:
            df['has_title'] = df['reviews.title'].notna() & (df['reviews.title'] != '')
            df['title_length'] = df['reviews.title'].apply(
                lambda x: len(str(x)) if pd.notnull(x) else 0
            )
        
        print(f"Feature extraction complete. DataFrame now has {df.shape[1]} columns.")
        return df
    
    def plot_feature_importance(self, feature_names, importances, title="Feature Importance"):
        """Plot feature importance
        
        Args:
            feature_names (list): Names of features
            importances (array): Importance scores
            title (str): Plot title
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        top_n = min(20, len(feature_names))  # Show top 20 features or less
        
        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(range(top_n), importances[indices][:top_n], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
        plt.xlim([-1, top_n])
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png')
        plt.show()

# Test function
def test_feature_engineering():
    """Test feature engineering functionality"""
    # Create sample data
    # 4_feature_engineering.py (continued)

    texts = [
        "This product is amazing! I love it so much.",
        "Terrible quality, broke after one day. Would not recommend!",
        "It's okay, not great but not bad either. Average product."
    ]
    
    df = pd.DataFrame({
        'processed_text': texts,
        'cleaned_text': texts,
        'reviews.text': texts,
        'reviews.rating': [5, 1, 3]
    })
    
    # Create feature engineer with adjusted parameters for small dataset
    fe = FeatureEngineer(max_features=100)
    
    # Fit TF-IDF
    fe.fit_tfidf(df['processed_text'])
    
    # Transform to TF-IDF
    tfidf_features = fe.transform_tfidf(df['processed_text'])
    print(f"TF-IDF matrix shape: {tfidf_features.shape}")
    
    # Extract additional features
    df = fe.extract_all_features(df)
    print("Extracted features:")
    print(df.columns.tolist())
    
    print("Feature engineering test completed!")
    return df

if __name__ == "__main__":
    test_feature_engineering()