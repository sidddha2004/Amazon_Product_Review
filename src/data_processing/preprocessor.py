import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """
    General text preprocessing utilities
    """
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """
        Clean and normalize text data
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Remove special characters and numbers (keep spaces and alphanumeric)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        """
        if not text:
            return []
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """
        Lemmatize tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text, remove_stops=True, lemmatize=True):
        """
        Complete preprocessing pipeline
        """
        # Clean text
        clean_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(clean_text)
        
        # Remove stopwords if requested
        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        
        # Lemmatize if requested
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return tokens
    
    def get_processed_text(self, text, remove_stops=True, lemmatize=True):
        """
        Get fully processed text as a string
        """
        tokens = self.preprocess_text(text, remove_stops, lemmatize)
        return ' '.join(tokens)