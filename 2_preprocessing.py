# 2_preprocessing.py - Text preprocessing for Amazon Review Sentiment Analysis

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """Class for preprocessing text data"""
    
    def __init__(self):
        """Initialize the text preprocessor with NLTK resources"""
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stopwords = set(stopwords.words('english'))
            
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean text by removing special characters, HTML tags, etc.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ''
        
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Insert space after periods that don't have spaces
        text = re.sub(r'\.(?=[a-zA-Z])', '. ', text)
        # Remove special characters and digits (keep apostrophes for contractions)
        text = re.sub(r'[^a-zA-Z\'\s]', ' ', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text after removing stopwords
        
        Args:
            text (str): Cleaned input text
            
        Returns:
            str: Processed text with tokens lemmatized and joined
        """
        if not text:
            return ''
            
        # Tokenize - with simple fallback if NLTK resources aren't available
        try:
            tokens = word_tokenize(text)
        except LookupError:
            # Simple tokenization as fallback
            tokens = text.split()
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]
        
        return ' '.join(tokens)
    
    def preprocess_text(self, text):
        """Run full preprocessing pipeline on text
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Fully preprocessed text
        """
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_lemmatize(cleaned)
        return processed

# Sentiment labeling function
def label_sentiment(rating, threshold_positive=4, threshold_negative=2):
    """Convert numeric rating to sentiment label
    
    Args:
        rating (float): Rating value (typically 1-5)
        threshold_positive (int): Min rating for positive sentiment
        threshold_negative (int): Max rating for negative sentiment
        
    Returns:
        str: Sentiment label ('positive', 'negative', or 'neutral')
    """
    if rating >= threshold_positive:
        return 'positive'
    elif rating <= threshold_negative:
        return 'negative'
    else:
        return 'neutral'

# Test function
def test_preprocessing():
    """Test the text preprocessing functionality"""
    processor = TextPreprocessor()
    
    # Test text
    test_text = "This is a <b>TEST</b> product!! It's working well. http://example.com"
    
    # Test clean_text
    cleaned = processor.clean_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test tokenize_and_lemmatize
    processed = processor.tokenize_and_lemmatize(cleaned)
    print(f"Processed: {processed}")
    
    # Test full pipeline
    full_processed = processor.preprocess_text(test_text)
    print(f"Full pipeline: {full_processed}")
    
    # Test sentiment labeling
    print(f"Rating 5 → {label_sentiment(5)}")
    print(f"Rating 3 → {label_sentiment(3)}")
    print(f"Rating 1 → {label_sentiment(1)}")

if __name__ == "__main__":
    test_preprocessing()
    