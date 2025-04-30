import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
from collections import Counter
import re

class FeatureEngineer:
    """
    Feature engineering for Amazon product reviews
    """
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.sia = SentimentIntensityAnalyzer()
    
    def extract_text_features(self, texts):
        """
        Extract features from review text
        """
        features = pd.DataFrame()
        
        # Text length features
        features['text_length'] = [len(str(text).split()) for text in texts]
        features['char_length'] = [len(str(text)) for text in texts]
        features['avg_word_length'] = [
            np.mean([len(w) for w in str(text).split()]) if len(str(text).split()) > 0 else 0 
            for text in texts
        ]
        
        # Sentence structure
        features['sentence_count'] = [
            len(self.nlp(str(text)).sents) for text in texts
        ]
        features['avg_sentence_length'] = features['text_length'] / features['sentence_count'].clip(lower=1)
        
        # Punctuation features
        features['question_mark_count'] = [str(text).count('?') for text in texts]
        features['exclamation_mark_count'] = [str(text).count('!') for text in texts]
        
        # Capitalization features
        features['all_caps_word_count'] = [
            sum(1 for word in str(text).split() if word.isupper() and len(word) > 1)
            for text in texts
        ]
        
        # Sentiment features
        sentiment_scores = [self.sia.polarity_scores(str(text)) for text in texts]
        features['sentiment_pos'] = [score['pos'] for score in sentiment_scores]
        features['sentiment_neg'] = [score['neg'] for score in sentiment_scores]
        features['sentiment_neu'] = [score['neu'] for score in sentiment_scores]
        features['sentiment_compound'] = [score['compound'] for score in sentiment_scores]
        
        return features
    
    def extract_part_of_speech_features(self, texts, top_n=10):
        """
        Extract part-of-speech features
        """
        pos_features = pd.DataFrame()
        
        # Process each text
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processing text {i}/{len(texts)}")
            
            if not isinstance(text, str) or not text.strip():
                continue
                
            doc = self.nlp(text)
            
            # Count POS tags
            pos_counts = Counter([token.pos_ for token in doc])
            
            # Add to features
            for pos, count in pos_counts.items():
                col_name = f'pos_{pos}'
                if col_name not in pos_features:
                    pos_features[col_name] = np.zeros(len(texts))
                pos_features.at[i, col_name] = count
        
        # Normalize by text length
        text_lengths = np.array([len(str(text).split()) for text in texts]).clip(min=1)
        for col in pos_features.columns:
            pos_features[col] = pos_features[col] / text_lengths
        
        return pos_features
    
    def extract_ngram_features(self, texts, n=2, top_k=100):
        """
        Extract top n-gram features
        """
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Create vectorizer for n-grams
        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            max_features=top_k,
            stop_words='english'
        )
        
        # Fit and transform
        ngram_counts = vectorizer.fit_transform([str(text) for text in texts])
        
        # Convert to DataFrame
        ngram_features = pd.DataFrame(
            ngram_counts.toarray(),
            columns=[f'ngram_{ngram}' for ngram in vectorizer.get_feature_names_out()]
        )
        
        return ngram_features
    
    def extract_product_features(self, df):
        """
        Extract features related to products
        """
        product_features = pd.DataFrame()
        
        # Price features (if available)
        if 'price' in df.columns:
            # Create price buckets
            df['price_bucket'] = pd.qcut(
                df['price'].clip(lower=0), 
                q=5, 
                labels=False, 
                duplicates='drop'
            )
            product_features['price_bucket'] = df['price_bucket']
        
        # Brand features (if available)
        if 'brand' in df.columns:
            # Count reviews per brand
            brand_counts = df['brand'].value_counts()
            
            # Get top brands (more than 10 reviews)
            top_brands = brand_counts[brand_counts >= 10].index
            
            # Group less common brands
            df['brand_grouped'] = df['brand'].apply(
                lambda x: x if x in top_brands else 'other'
            )
            
            # One-hot encode brands
            brand_dummies = pd.get_dummies(df['brand_grouped'], prefix='brand')
            product_features = pd.concat([product_features, brand_dummies], axis=1)
        
        # Category features (if available)
        if 'categories' in df.columns:
            # Extract primary category
            df['primary_category'] = df['categories'].apply(
                lambda x: str(x).split(',')[0].strip() if isinstance(x, str) else 'unknown'
            )
            
            # Count reviews per category
            category_counts = df['primary_category'].value_counts()
            
            # Get top categories
            top_categories = category_counts[category_counts >= 10].index
            
            # Group less common categories
            df['category_grouped'] = df['primary_category'].apply(
                lambda x: x if x in top_categories else 'other'
            )
            
            # One-hot encode categories
            category_dummies = pd.get_dummies(df['category_grouped'], prefix='category')
            product_features = pd.concat([product_features, category_dummies], axis=1)
        
        return product_features
    
    def combine_features(self, *feature_dfs):
        """
        Combine multiple feature DataFrames
        """
        combined = pd.concat(feature_dfs, axis=1)
        
        # Fill any missing values
        combined = combined.fillna(0)
        
        return combined