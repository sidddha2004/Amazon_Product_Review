# 3_data_processing.py - Data loading and processing for Amazon Review Sentiment Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import time
import gc
from wordcloud import WordCloud

# We'll now define the TextPreprocessor and label_sentiment in case they're not imported
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
        if not isinstance(text, str):
            return ''
        
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\.(?=[a-zA-Z])', '. ', text)
        text = re.sub(r'[^a-zA-Z\'\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        if not text:
            return ''
            
        try:
            tokens = word_tokenize(text)
        except LookupError:
            tokens = text.split()
        
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]
        
        return ' '.join(tokens)
    
    def preprocess_text(self, text):
        cleaned = self.clean_text(text)
        processed = self.tokenize_and_lemmatize(cleaned)
        return processed

def label_sentiment(rating, threshold_positive=4, threshold_negative=2):
    if rating >= threshold_positive:
        return 'positive'
    elif rating <= threshold_negative:
        return 'negative'
    else:
        return 'neutral'

class DataProcessor:
    """Class for loading and processing Amazon review data"""
    
    def __init__(self, text_preprocessor=None):
        """Initialize with a text preprocessor
        
        Args:
            text_preprocessor: Instance of TextPreprocessor (optional)
        """
        self.text_preprocessor = text_preprocessor if text_preprocessor else TextPreprocessor()
    
    def load_data(self, filepath, chunk_size=10000):
        """Load large CSV in chunks to save memory
        
        Args:
            filepath (str): Path to the CSV file
            chunk_size (int): Size of chunks for reading large files
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        print(f"Loading data from {filepath}...")
        
        try:
            # For smaller files or testing
            df = pd.read_csv(filepath)
            print(f"Loaded dataset with {len(df)} rows")
            return df
        except:
            # For larger files, use chunking
            chunks = []
            for chunk in tqdm(pd.read_csv(filepath, chunksize=chunk_size), 
                            desc="Loading data chunks"):
                chunks.append(chunk)
                
            df = pd.concat(chunks)
            print(f"Loaded dataset with {len(df)} rows using {len(chunks)} chunks")
            return df
    
    def preprocess_data(self, df, sample_size=None, random_state=42):
        """Preprocess the dataframe with optional sampling
        
        Args:
            df (pd.DataFrame): Dataframe to preprocess
            sample_size (int): Optional sample size to reduce dataset
            random_state (int): Random state for reproducibility
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        start_time = time.time()
        print("Starting data preprocessing...")
        
        # Sample if requested (for faster development/testing)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_state)
            print(f"Sampled {sample_size} rows from dataset")
        
        # Handle missing values
        df = df.fillna({
            'reviews.rating': 0,
            'reviews.text': '',
            'reviews.title': ''
        })
        
        # Convert ratings to float
        df['reviews.rating'] = pd.to_numeric(df['reviews.rating'], errors='coerce')
        
        # Create sentiment labels based on rating
        df['sentiment'] = df['reviews.rating'].apply(label_sentiment)
        
        # Process text in batches to save memory
        tqdm.pandas(desc="Cleaning review text")
        df['cleaned_text'] = df['reviews.text'].progress_apply(self.text_preprocessor.clean_text)
        
        # Optional: Add title to the text if available
        df['has_title'] = df['reviews.title'].notna() & (df['reviews.title'] != '')
        df.loc[df['has_title'], 'cleaned_text'] = df.loc[df['has_title'], 'reviews.title'] + ". " + df.loc[df['has_title'], 'cleaned_text']
        
        # Create a lemmatized version for feature engineering (processing in batches)
        tqdm.pandas(desc="Lemmatizing text")
        df['processed_text'] = df['cleaned_text'].progress_apply(self.text_preprocessor.tokenize_and_lemmatize)
        
        # Drop rows with empty processed text
        df = df[df['processed_text'] != '']
        
        # Add text length features
        df['text_length'] = df['cleaned_text'].apply(len)
        df['word_count'] = df['cleaned_text'].apply(lambda x: len(x.split()))
        
        print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        # Free memory
        gc.collect()
        
        return df
    
    def create_train_val_test_split(self, df, test_size=0.2, val_size=0.25, random_state=42):
        """Create stratified train/validation/test splits
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            test_size (float): Proportion of data for testing (0-1)
            val_size (float): Proportion of test data for validation (0-1)
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        print("Creating data splits...")
        
        # Check if dataset is too small for default parameters
        min_samples_per_class = df['sentiment'].value_counts().min()
        n_classes = len(df['sentiment'].unique())
        total_samples = len(df)
        
        # For very small datasets, adjust approach
        if total_samples < n_classes * 4:
            print("Warning: Dataset too small for stratified split. Using simple random split.")
            train_size = 0.6
            val_size = 0.2
            test_size = 0.2
            
            # Random sampling
            train_df = df.sample(frac=train_size, random_state=random_state)
            temp_df = df.drop(train_df.index)
            val_df = temp_df.sample(frac=val_size/(val_size+test_size), random_state=random_state)
            test_df = temp_df.drop(val_df.index)
        else:
            # Adjust test_size if needed for stratification
            if int(total_samples * test_size) < n_classes:
                adjusted_test_size = n_classes / total_samples
                print(f"Warning: Adjusting test_size from {test_size} to {adjusted_test_size} to ensure at least one sample per class")
                test_size = adjusted_test_size
            
            # First split: training and temp (validation + test)
            train_df, temp_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=df['sentiment']
            )
            
            # For very small datasets, just do a train/test split
            if len(temp_df) < 2 * n_classes:
                print("Warning: Dataset too small for three-way split. Using train/test split only.")
                test_df = temp_df
                val_df = pd.DataFrame(columns=train_df.columns)  # Empty dataframe with same columns
            else:
                # Second split: validation and test from temp
                val_df, test_df = train_test_split(
                    temp_df, 
                    test_size=val_size, 
                    random_state=random_state, 
                    stratify=temp_df['sentiment']
                )
        
        print(f"Training set: {len(train_df)} rows")
        print(f"Validation set: {len(val_df)} rows")
        print(f"Test set: {len(test_df)} rows")
        
        return train_df, val_df, test_df
    
    def analyze_dataset(self, df):
        """Generate dataset insights and visualizations
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
        """
        print("\n=== Dataset Analysis ===")
        print(f"Total reviews: {len(df)}")
        sentiment_counts = df['sentiment'].value_counts()
        print(f"Sentiment distribution: {sentiment_counts.to_dict()}")
        
        # Plot sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment', data=df, palette='viridis')
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('visualizations/sentiment_distribution.png')
        plt.show()
        
        # Review length analysis
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='sentiment', y='word_count', data=df)
        plt.title('Review Word Count by Sentiment')
        plt.xlabel('Sentiment')
        plt.ylabel('Word Count')
        plt.tight_layout()
        plt.savefig('visualizations/word_count_by_sentiment.png')
        plt.show()
        
        # Word clouds by sentiment
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        sentiments = ['positive', 'neutral', 'negative']
        
        for i, sentiment in enumerate(sentiments):
            # Check if sentiment exists in the dataframe
            if sentiment not in df['sentiment'].values:
                axes[i].text(0.5, 0.5, f"No {sentiment} reviews in dataset", 
                            horizontalalignment='center', verticalalignment='center')
                axes[i].set_title(f'{sentiment.capitalize()} Reviews (None Found)')
                axes[i].axis('off')
                continue
                
            text = ' '.join(df[df['sentiment'] == sentiment]['processed_text'])
            if not text.strip():
                axes[i].text(0.5, 0.5, f"No text found for {sentiment} reviews", 
                            horizontalalignment='center', verticalalignment='center')
                axes[i].set_title(f'{sentiment.capitalize()} Reviews (No Text)')
                axes[i].axis('off')
                continue
                
            wordcloud = WordCloud(width=800, height=400, background_color='white',
                                max_words=100, contour_width=3).generate(text)
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{sentiment.capitalize()} Reviews')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/sentiment_wordclouds.png')
        plt.show()
        
        # Basic statistics about ratings
        print("\nRating Statistics:")
        print(df['reviews.rating'].describe())
        
        # Product category analysis if available
        if 'categories' in df.columns:
            print("\nTop Product Categories:")
            categories = df['categories'].str.split(',', expand=True).stack()
            print(categories.value_counts().head(10))

# Test function as a standalone function, not a method of DataProcessor
def test_data_processing():
    """Test data processing functionality with a small sample CSV"""
    # Create dummy data for testing
    data = {
        'reviews.text': [
            "This product is amazing! I love it.",
            "Great purchase, would buy again!",
            "Excellent quality and fast shipping.",
            "Not worth the money, it broke after 2 days.",
            "Terrible product, avoid at all costs.",
            "Poor quality and overpriced.",
            "It's okay, nothing special.",
            "Average product, does what it says.",
            "Neither great nor terrible, just average."
        ],
        'reviews.title': [
            "Great Purchase",
            "Highly Recommend",
            "Five Stars",
            "Waste of Money",
            "Disappointed",
            "Don't Buy This",
            "Average Product",
            "It's Alright",
            "Middle of the Road"
        ],
        'reviews.rating': [5, 5, 5, 1, 1, 1, 3, 3, 3]
    }
    
    test_df = pd.DataFrame(data)
    
    # Save to CSV for testing
    test_df.to_csv('test_reviews.csv', index=False)
    
    # Process the test data
    processor = DataProcessor()
    df = processor.load_data('test_reviews.csv')
    processed_df = processor.preprocess_data(df)
    
    # For testing with a tiny dataset, use custom split ratios
    # We need at least 3 samples in test (one per class)
    train_size = 3/9  # Keep 3 samples for training
    test_size = 6/9   # Use 6 samples for temp (test+val)
    
    # First split
    train_df, temp_df = train_test_split(
        processed_df, 
        test_size=test_size, 
        random_state=42, 
        stratify=processed_df['sentiment']
    )
    
    # Second split - equal size for val and test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['sentiment']
    )
    
    print(f"Training set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    print("Data processing test completed!")
    return train_df, val_df, test_df

if __name__ == "__main__":
    test_data_processing()