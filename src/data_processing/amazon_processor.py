import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
from .preprocessor import TextPreprocessor

class AmazonDataProcessor:
    """
    Processor for the Amazon product reviews dataset from Datafiniti
    """
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.text_preprocessor = TextPreprocessor()
        
        # Create directories if they don't exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_data(self, filename):
        """
        Load Amazon review data from CSV file
        """
        file_path = os.path.join(self.raw_dir, filename)
        print(f"Attempting to load data from: {file_path}")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            
            # Check if it's a path issue - try with just the filename
            basename = os.path.basename(filename)
            alt_path = os.path.join(self.raw_dir, basename)
            if os.path.exists(alt_path):
                print(f"Found file at alternative path: {alt_path}")
                file_path = alt_path
            else:
                # Try the direct path as given
                if os.path.exists(filename):
                    print(f"Using direct path: {filename}")
                    file_path = filename
                else:
                    print(f"Could not find file. Checked paths:")
                    print(f"  - {file_path}")
                    print(f"  - {alt_path}")
                    print(f"  - {filename}")
                    return None
    
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_dataset(self, df):
        """
        Clean and preprocess the raw dataframe

        """
        if df is None:
            print("Error: DataFrame is None, cannot clean dataset")
            return None
        # Make a copy to avoid modifying the original
        clean_df = df.copy()
        
        # Handle missing values
        clean_df = clean_df.fillna({
            'reviews.rating': 0,
            'reviews.text': '',
            'reviews.title': '',
            'reviews.username': 'anonymous',
            'reviews.doRecommend': False
        })
        
        # Convert ratings to float
        clean_df['reviews.rating'] = pd.to_numeric(clean_df['reviews.rating'], errors='coerce')
        
        # Convert dates
        date_columns = [col for col in clean_df.columns if '.date' in col.lower()]
        for col in date_columns:
            clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce')
        
        # Extract review year and month as separate features
        if 'reviews.date' in clean_df.columns:
            clean_df['review_year'] = clean_df['reviews.date'].dt.year
            clean_df['review_month'] = clean_df['reviews.date'].dt.month
        
        # Process product categories
        if 'categories' in clean_df.columns:
            # Categories are comma-separated strings
            clean_df['categories'] = clean_df['categories'].fillna('')
            clean_df['category_list'] = clean_df['categories'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
            clean_df['category_count'] = clean_df['category_list'].apply(len)
            
            # Extract primary category (first in the list)
            clean_df['primary_category'] = clean_df['category_list'].apply(lambda x: x[0].strip() if len(x) > 0 else 'Unknown')
        
        # Process text fields
        text_cols = ['reviews.text', 'reviews.title']
        for col in text_cols:
            if col in clean_df.columns:
                # Clean text
                clean_df[f'cleaned_{col}'] = clean_df[col].apply(self.text_preprocessor.clean_text)
                
                # Calculate text length
                clean_df[f'{col}_length'] = clean_df[col].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        
        # Create binary sentiment labels based on rating
        if 'reviews.rating' in clean_df.columns:
            clean_df['sentiment'] = clean_df['reviews.rating'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
            )
        
        return clean_df
    
    def extract_product_features(self, df):
        """
        Extract product-specific features from the dataset
        """
        product_df = df.copy()
        
        # Group by product ID (assuming 'id' or 'asins' is the product identifier)
        product_id = 'asins' if 'asins' in product_df.columns else 'id'
        
        # Aggregate reviews by product
        product_stats = product_df.groupby(product_id).agg({
            'reviews.rating': ['count', 'mean', 'std', 'min', 'max'],
            'reviews.text_length': ['mean', 'max'],
            'sentiment': lambda x: x.value_counts().to_dict()
        })
        
        # Flatten multi-level column names
        product_stats.columns = ['_'.join(col).strip() for col in product_stats.columns.values]
        
        # Rename columns for clarity
        product_stats = product_stats.rename(columns={
            'reviews.rating_count': 'review_count',
            'reviews.rating_mean': 'average_rating',
            'reviews.rating_std': 'rating_std_dev',
            'reviews.rating_min': 'min_rating',
            'reviews.rating_max': 'max_rating',
            'reviews.text_length_mean': 'avg_review_length',
            'reviews.text_length_max': 'max_review_length',
            'sentiment_<lambda>': 'sentiment_counts'
        })
        
        # Calculate rating distribution (percentage of 1-5 stars)
        for rating in range(1, 6):
            product_stats[f'pct_{rating}star'] = product_df[product_df['reviews.rating'] == rating].groupby(product_id).size() / product_stats['review_count']
        
        # Fill NaN values with 0 for the percentage columns
        for col in [col for col in product_stats.columns if 'pct_' in col]:
            product_stats[col] = product_stats[col].fillna(0)
        
        return product_stats
    
    def prepare_model_data(self, df):
        """
        Prepare data specifically for model training
        """
        if df is None:
            print("Error: DataFrame is None, cannot prepare model data")
            return None
            
        # Create a copy with selected columns
        columns_to_include = []
        
        # Review text
        if 'reviews.text' in df.columns:
            columns_to_include.append('reviews.text')
        if 'cleaned_reviews.text' in df.columns:
            columns_to_include.append('cleaned_reviews.text')
            
        # Review metadata
        if 'reviews.rating' in df.columns:
            columns_to_include.append('reviews.rating')
        if 'sentiment' in df.columns:
            columns_to_include.append('sentiment')
        if 'reviews.title' in df.columns:
            columns_to_include.append('reviews.title')
            
        # Product metadata
        if 'brand' in df.columns:
            columns_to_include.append('brand')
        if 'primary_category' in df.columns:
            columns_to_include.append('primary_category')
            
        # Use all columns if our list is empty (shouldn't happen)
        if not columns_to_include:
            print("Warning: No expected columns found. Using all columns.")
            model_df = df.copy()
        else:
            model_df = df[columns_to_include].copy()
        
        # Rename columns for clarity and consistency
        column_mapping = {
            'reviews.text': 'review_text',
            'cleaned_reviews.text': 'cleaned_text',
            'reviews.rating': 'rating',
            'reviews.title': 'review_title',
            'primary_category': 'category'
        }
        
        # Only rename columns that exist
        for old_col, new_col in column_mapping.items():
            if old_col in model_df.columns:
                model_df = model_df.rename(columns={old_col: new_col})
        
        # Drop rows with missing critical values
        critical_columns = []
        if 'review_text' in model_df.columns:
            critical_columns.append('review_text')
        elif 'cleaned_text' in model_df.columns:
            critical_columns.append('cleaned_text')
            
        if 'rating' in model_df.columns:
            critical_columns.append('rating')
            
        if critical_columns:
            model_df = model_df.dropna(subset=critical_columns)
        
        # Ensure we have sentiment column
        if 'sentiment' not in model_df.columns and 'rating' in model_df.columns:
            model_df['sentiment'] = model_df['rating'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
            )
        
        # Ensure we have cleaned text column
        if 'cleaned_text' not in model_df.columns and 'review_text' in model_df.columns:
            model_df['cleaned_text'] = model_df['review_text'].apply(self.clean_text)
        
        return model_df
    
    def split_data(self, df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Split data into training, validation, and test sets
        """
        from sklearn.model_selection import train_test_split
        
        if df is None:
            print("Error: DataFrame is None, cannot split data")
            return None, None, None
            
        # Ensure the proportions add up to 1
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split proportions must sum to 1"
        
        # First split into train and temp sets
        try:
            if 'sentiment' in df.columns:
                train_df, temp_df = train_test_split(
                    df, 
                    train_size=train_size,
                    random_state=random_state,
                    stratify=df['sentiment']
                )
            else:
                train_df, temp_df = train_test_split(
                    df, 
                    train_size=train_size,
                    random_state=random_state
                )
            
            # Then split temp into validation and test sets
            relative_val_size = val_size / (val_size + test_size)
            
            if 'sentiment' in temp_df.columns:
                val_df, test_df = train_test_split(
                    temp_df,
                    train_size=relative_val_size,
                    random_state=random_state,
                    stratify=temp_df['sentiment']
                )
            else:
                val_df, test_df = train_test_split(
                    temp_df,
                    train_size=relative_val_size,
                    random_state=random_state
                )
            
            print(f"Data split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
            return train_df, val_df, test_df
            
        except Exception as e:
            print(f"Error splitting data: {e}")
            # Fallback to simple splitting without stratification
            try:
                train_df, temp_df = train_test_split(df, train_size=train_size, random_state=random_state)
                val_df, test_df = train_test_split(temp_df, train_size=0.5, random_state=random_state)
                print(f"Data split (fallback method): {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
                return train_df, val_df, test_df
            except Exception as e2:
                print(f"Error in fallback split: {e2}")
                return None, None, None
    
    def save_processed_data(self, df, filename):
        """
        Save processed dataframe to CSV
        """
        if df is None:
            print("Error: DataFrame is None, cannot save processed data")
            return
            
        output_path = os.path.join(self.processed_dir, filename)
        try:
            df.to_csv(output_path, index=False)
            print(f"Saved processed data to {output_path}")
        except Exception as e:
            print(f"Error saving processed data: {e}")
    
    def process_pipeline(self, input_filename, output_filename='processed_amazon_reviews.csv'):
        """
        Run the complete data processing pipeline
        """
        print("Starting data processing pipeline...")
        
        # Load raw data
        raw_df = self.load_data(input_filename)
        if raw_df is None:
            return None
        
        # Clean and preprocess data
        print("Cleaning and preprocessing data...")
        clean_df = self.clean_dataset(raw_df)
        
        # Extract product features
        print("Extracting product features...")
        product_features = self.extract_product_features(clean_df)
        
        # Prepare data for modeling
        print("Preparing data for modeling...")
        model_df = self.prepare_model_data(clean_df)
        
        # Save processed data
        self.save_processed_data(model_df, output_filename)
        self.save_processed_data(product_features, 'product_features.csv')
        
        print("Data processing completed successfully!")
        return model_df, product_features