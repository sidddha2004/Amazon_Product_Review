import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.amazon_processor import AmazonDataProcessor
from src.models.sentiment_model import SentimentAnalysisModel
from src.models.aspect_model import AspectBasedSentimentModel
from src.models.helpfulness_model import ReviewHelpfulnessModel
from src.models.summarizer_model import ReviewSummarizerModel
from src.features.feature_engineering import FeatureEngineer

def train_sentiment_model(train_df, val_df, test_df, model_dir):
    """
    Train and evaluate the sentiment analysis model
    """
    print("\n" + "="*50)
    print("Training Sentiment Analysis Model")
    print("="*50)
    
    # Check column names
    print("Available columns:", train_df.columns.tolist())
    
    # Determine the correct column name for review text
    if 'cleaned_text' in train_df.columns:
        text_column = 'cleaned_text'
    elif 'cleaned_reviews.text' in train_df.columns:
        text_column = 'cleaned_reviews.text'
    elif 'review_text' in train_df.columns:
        text_column = 'review_text'
    elif 'reviews.text' in train_df.columns:
        text_column = 'reviews.text'
    else:
        # If none of the expected columns are found, use the first text-like column
        text_columns = [col for col in train_df.columns if 'text' in col.lower()]
        if text_columns:
            text_column = text_columns[0]
            print(f"Using {text_column} as the text column")
        else:
            raise ValueError("Could not find a suitable text column in the data")
    
    # Determine the correct column name for sentiment
    if 'sentiment' in train_df.columns:
        sentiment_column = 'sentiment'
    elif 'sentiment_label' in train_df.columns:
        sentiment_column = 'sentiment_label'
    else:
        # If sentiment column not found, create one based on rating
        if 'rating' in train_df.columns:
            print("Creating sentiment column from rating")
            train_df['sentiment'] = train_df['rating'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
            val_df['sentiment'] = val_df['rating'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
            test_df['sentiment'] = test_df['rating'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
            sentiment_column = 'sentiment'
        elif 'reviews.rating' in train_df.columns:
            print("Creating sentiment column from reviews.rating")
            train_df['sentiment'] = train_df['reviews.rating'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
            val_df['sentiment'] = val_df['reviews.rating'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
            test_df['sentiment'] = test_df['reviews.rating'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral'))
            sentiment_column = 'sentiment'
        else:
            raise ValueError("Could not find a suitable sentiment or rating column in the data")
    
    # Initialize model
    sentiment_model = SentimentAnalysisModel(
        max_words=15000,
        max_sequence_length=250,
        embedding_dim=100
    )
    
    # Prepare data - Fixed for unpacking issue
    result = sentiment_model.preprocess_data(
        train_df[text_column].fillna('').values,
        train_df[sentiment_column].values,
        [], []
    )
    
    # Handle different return values from preprocess_data
    if isinstance(result, tuple):
        if len(result) == 4:
            X_train, y_train, _, _ = result
        elif len(result) == 2:
            X_train, y_train = result
        else:
            print(f"Warning: preprocess_data returned {len(result)} values instead of expected 2 or 4")
            X_train = result[0]
            y_train = result[1]
    else:
        # If result is not a tuple, handle accordingly
        print(f"Warning: preprocess_data returned a {type(result)} instead of a tuple")
        X_train = result
        y_train = np.zeros(len(X_train))  # Fallback
    
    # Same for validation data
    result_val = sentiment_model.preprocess_data(
        val_df[text_column].fillna('').values,
        val_df[sentiment_column].values,
        [], []
    )
    
    if isinstance(result_val, tuple):
        if len(result_val) == 4:
            X_val, y_val, _, _ = result_val
        elif len(result_val) == 2:
            X_val, y_val = result_val
        else:
            X_val = result_val[0]
            y_val = result_val[1]
    else:
        X_val = result_val
        y_val = np.zeros(len(X_val))  # Fallback
    
    # Same for test data
    result_test = sentiment_model.preprocess_data(
        test_df[text_column].fillna('').values,
        test_df[sentiment_column].values,
        [], []
    )
    
    if isinstance(result_test, tuple):
        if len(result_test) == 4:
            X_test, y_test, _, _ = result_test
        elif len(result_test) == 2:
            X_test, y_test = result_test
        else:
            X_test = result_test[0]
            y_test = result_test[1]
    else:
        X_test = result_test
        y_test = np.zeros(len(X_test))  # Fallback
    
    # Build and train model
# Fix this in train_models.py
    if len(y_train.shape) > 1:
        # Already one-hot encoded
        num_classes = y_train.shape[1]
    else:
        # Not yet one-hot encoded
        num_classes = len(np.unique(y_train))

    print(f"Training with {num_classes} sentiment classes")    
    print(f"Training with {num_classes} sentiment classes")
    
    model = sentiment_model.build_cnn_lstm_model(num_classes)
    print(model.summary())
    
    # Train model
    history = sentiment_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=64
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save model
    os.makedirs(model_dir, exist_ok=True)
    sentiment_model.save_model(
        os.path.join(model_dir, 'sentiment_model'),
        os.path.join(model_dir, 'sentiment_tokenizer.pkl')
    )
    
    return sentiment_model, history

def train_aspect_model(train_df, val_df, test_df, model_dir):
    """
    Train and evaluate the aspect-based sentiment model
    """
    print("\n" + "="*50)
    print("Training Aspect-Based Sentiment Model")
    print("="*50)
    
    # Determine the correct column name for review text
    if 'cleaned_text' in train_df.columns:
        text_column = 'cleaned_text'
    elif 'cleaned_reviews.text' in train_df.columns:
        text_column = 'cleaned_reviews.text'
    elif 'review_text' in train_df.columns:
        text_column = 'review_text'
    elif 'reviews.text' in train_df.columns:
        text_column = 'reviews.text'
    else:
        # If none of the expected columns are found, use the first text-like column
        text_columns = [col for col in train_df.columns if 'text' in col.lower()]
        if text_columns:
            text_column = text_columns[0]
            print(f"Using {text_column} as the text column")
        else:
            raise ValueError("Could not find a suitable text column in the data")
    
    # Determine the correct column name for sentiment
    if 'sentiment' in train_df.columns:
        sentiment_column = 'sentiment'
    elif 'sentiment_label' in train_df.columns:
        sentiment_column = 'sentiment_label'
    else:
        # Use the one created in the sentiment model training function
        sentiment_column = 'sentiment'
        
    # Initialize model
    aspect_model = AspectBasedSentimentModel(
        max_words=15000,
        max_sequence_length=250,
        embedding_dim=100
    )
    
    # Extract common aspects from reviews
    common_aspects = aspect_model.extract_aspects(
        train_df[text_column].fillna('').values,
        num_aspects=15
    )
    print(f"Extracted aspects: {common_aspects}")
    
    # Create aspect-specific training data
    try:
        train_texts, train_aspects, train_labels = aspect_model.create_aspect_training_data(
            train_df[text_column].fillna('').values,
            train_df[sentiment_column].values,
            aspects=common_aspects
        )
        
        # Split into train/val
        texts_train, texts_val, aspects_train, aspects_val, labels_train, labels_val = train_test_split(
            train_texts, train_aspects, train_labels, test_size=0.2, random_state=42
        )
        
        # Preprocess data
        X_train, y_train = aspect_model.preprocess_data(texts_train, aspects_train, labels_train)
        X_val, y_val = aspect_model.preprocess_data(texts_val, aspects_val, labels_val)
        
        # Build and train model
        num_classes = len(np.unique(labels_train))
        print(f"Training with {num_classes} sentiment classes")
        
        model = aspect_model.build_model(num_classes)
        print(model.summary())
        
        # Train model
        history = aspect_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=3,
            batch_size=32
        )
        
        # Save model
        os.makedirs(model_dir, exist_ok=True)
        aspect_model.save_model(
            os.path.join(model_dir, 'aspect_model'),
            os.path.join(model_dir, 'aspect_tokenizer.pkl')
        )
        
        return aspect_model, history
    
    except Exception as e:
        print(f"Error in training aspect model: {e}")
        return None, None

def train_helpfulness_model(train_df, val_df, test_df, model_dir):
    """
    Train and evaluate the review helpfulness model
    """
    print("\n" + "="*50)
    print("Training Review Helpfulness Model")
    print("="*50)
    
    # Determine text column
    if 'review_text' in train_df.columns:
        text_column = 'review_text'
    elif 'reviews.text' in train_df.columns:
        text_column = 'reviews.text'
    elif 'cleaned_text' in train_df.columns:
        text_column = 'cleaned_text'
    else:
        text_columns = [col for col in train_df.columns if 'text' in col.lower()]
        if text_columns:
            text_column = text_columns[0]
        else:
            raise ValueError("Could not find a suitable text column for helpfulness model")
    
    # Ensure we have the necessary columns for the helpfulness model
    if 'review_title' not in train_df.columns and 'reviews.title' in train_df.columns:
        train_df['review_title'] = train_df['reviews.title']
        val_df['review_title'] = val_df['reviews.title']
        test_df['review_title'] = test_df['reviews.title']
    elif 'review_title' not in train_df.columns:
        train_df['review_title'] = ''
        val_df['review_title'] = ''
        test_df['review_title'] = ''
    
    if 'rating' not in train_df.columns and 'reviews.rating' in train_df.columns:
        train_df['rating'] = train_df['reviews.rating']
        val_df['rating'] = val_df['reviews.rating']
        test_df['rating'] = test_df['reviews.rating']
    elif 'rating' not in train_df.columns:
        train_df['rating'] = 0  # Default value
        val_df['rating'] = 0
        test_df['rating'] = 0
    
    # Rename text column if needed
    if text_column != 'review_text':
        train_df['review_text'] = train_df[text_column]
        val_df['review_text'] = val_df[text_column]
        test_df['review_text'] = test_df[text_column]
    
    # Initialize model
    helpfulness_model = ReviewHelpfulnessModel(model_type='gradient_boosting')
    
    try:
        # Extract features
        train_features = helpfulness_model.extract_features(train_df)
        val_features = helpfulness_model.extract_features(val_df)
        test_features = helpfulness_model.extract_features(test_df)
        
        # Create helpfulness labels
        train_labels = helpfulness_model.create_labels(train_df)
        val_labels = helpfulness_model.create_labels(val_df)
        test_labels = helpfulness_model.create_labels(test_df)
        
        print(f"Training with {train_features.shape[1]} features")
        print(f"Class distribution: {np.bincount(train_labels)}")
        
        # Train model
        helpfulness_model.train(train_features, train_labels)
        
        # Evaluate on validation set
        val_metrics = helpfulness_model.evaluate(val_features, val_labels)
        print("\nValidation Metrics:")
        print(f"Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"F1 Score: {val_metrics['f1_score']:.4f}")
        
        # Evaluate on test set
        test_metrics = helpfulness_model.evaluate(test_features, test_labels)
        print("\nTest Metrics:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Score: {test_metrics['f1_score']:.4f}")
        
        # Save model
        os.makedirs(model_dir, exist_ok=True)
        helpfulness_model.save_model(
            os.path.join(model_dir, 'helpfulness_model.pkl')
        )
        
        return helpfulness_model, test_metrics
    
    except Exception as e:
        print(f"Error in training helpfulness model: {e}")
        return None, None

def fine_tune_summarizer(train_df, val_df, model_dir, sample_size=1000):
    """
    Fine-tune the review summarization model
    """
    print("\n" + "="*50)
    print("Fine-tuning Review Summarizer Model")
    print("="*50)
    
    try:
        # Determine text column
        if 'review_text' in train_df.columns:
            text_column = 'review_text'
        elif 'reviews.text' in train_df.columns:
            text_column = 'reviews.text'
        elif 'cleaned_text' in train_df.columns:
            text_column = 'cleaned_text'
        else:
            text_columns = [col for col in train_df.columns if 'text' in col.lower()]
            if text_columns:
                text_column = text_columns[0]
            else:
                raise ValueError("Could not find a suitable text column for summarizer model")
        
        # Initialize model
        summarizer = ReviewSummarizerModel(
            model_name='t5-small',
            max_input_length=512,
            max_output_length=100
        )
        
        # Use only reviews with sufficient length
        # Add a length column if it doesn't exist
        if text_column + '_length' not in train_df.columns:
            train_df[text_column + '_length'] = train_df[text_column].apply(
                lambda x: len(str(x).split()) if isinstance(x, str) else 0
            )
            val_df[text_column + '_length'] = val_df[text_column].apply(
                lambda x: len(str(x).split()) if isinstance(x, str) else 0
            )
        
        length_column = text_column + '_length'
        
        train_reviews = train_df[train_df[length_column] > 50][text_column].fillna('').values
        val_reviews = val_df[val_df[length_column] > 50][text_column].fillna('').values
        
        # Limit to sample size to speed up fine-tuning
        train_reviews = train_reviews[:min(sample_size, len(train_reviews))]
        val_reviews = val_reviews[:min(200, len(val_reviews))]
        
        # Create extractive summaries as targets for fine-tuning
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        def extract_first_sentences(text, num_sentences=2):
            if not isinstance(text, str) or not text.strip():
                return ""
            sentences = nltk.sent_tokenize(text)
            return ' '.join(sentences[:num_sentences])
        
        train_summaries = [extract_first_sentences(review) for review in train_reviews]
        val_summaries = [extract_first_sentences(review) for review in val_reviews]
        
        # Fine-tune the model
        history = summarizer.fine_tune(
            train_reviews,
            train_summaries,
            validation_texts=val_reviews,
            validation_summaries=val_summaries,
            epochs=2,
            batch_size=4
        )
        
        # Save model
        os.makedirs(model_dir, exist_ok=True)
        summarizer.save_model(os.path.join(model_dir, 'summarizer_model'))
        
        return summarizer, history
    
    except Exception as e:
        print(f"Error in fine-tuning summarizer model: {e}")
        print("Skipping summarizer model training.")
        return None, None

def main():
    """
    Main function to run the training pipeline
    """
    parser = argparse.ArgumentParser(description='Train Amazon review analysis models')
    parser.add_argument('--data_file', type=str, required=True, help='Path to Amazon reviews CSV file')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory for models')
    parser.add_argument('--train_sentiment', action='store_true', help='Train sentiment analysis model')
    parser.add_argument('--train_aspect', action='store_true', help='Train aspect-based sentiment model')
    parser.add_argument('--train_helpfulness', action='store_true', help='Train review helpfulness model')
    parser.add_argument('--train_summarizer', action='store_true', help='Fine-tune review summarizer model')
    parser.add_argument('--train_all', action='store_true', help='Train all models')
    
    args = parser.parse_args()
    
    # If no specific model is specified, train all
    if not any([args.train_sentiment, args.train_aspect, args.train_helpfulness, args.train_summarizer]):
        args.train_all = True
    
    # Set all flags to True if train_all is specified
    if args.train_all:
        args.train_sentiment = True
        args.train_aspect = True
        args.train_helpfulness = True
        args.train_summarizer = True
    
    # Process data
    print("\nProcessing Amazon Reviews Data...")
    data_processor = AmazonDataProcessor()
    
    # Load data with better error handling
    raw_df = data_processor.load_data(args.data_file)
    if raw_df is None:
        print(f"Error: Could not load data from {args.data_file}")
        print("Please check if the file exists and is in the correct format.")
        return
    
    try:
        # Clean and preprocess data
        clean_df = data_processor.clean_dataset(raw_df)
        model_df = data_processor.prepare_model_data(clean_df)
        
        # Split data
        train_df, val_df, test_df = data_processor.split_data(model_df)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Train models as specified
        models_trained = 0
        
        if args.train_sentiment:
            try:
                print("\nTraining sentiment model...")
                sentiment_model, _ = train_sentiment_model(train_df, val_df, test_df, args.output_dir)
                if sentiment_model is not None:
                    models_trained += 1
            except Exception as e:
                print(f"Error training sentiment model: {e}")
                import traceback
                traceback.print_exc()
        
        if args.train_aspect:
            try:
                print("\nTraining aspect model...")
                aspect_model, _ = train_aspect_model(train_df, val_df, test_df, args.output_dir)
                if aspect_model is not None:
                    models_trained += 1
            except Exception as e:
                print(f"Error training aspect model: {e}")
                import traceback
                traceback.print_exc()
        
        if args.train_helpfulness:
            try:
                print("\nTraining helpfulness model...")
                helpfulness_model, _ = train_helpfulness_model(train_df, val_df, test_df, args.output_dir)
                if helpfulness_model is not None:
                    models_trained += 1
            except Exception as e:
                print(f"Error training helpfulness model: {e}")
                import traceback
                traceback.print_exc()
        
        if args.train_summarizer:
            try:
                print("\nFine-tuning summarizer model...")
                summarizer_model, _ = fine_tune_summarizer(train_df, val_df, args.output_dir)
                if summarizer_model is not None:
                    models_trained += 1
            except Exception as e:
                print(f"Error fine-tuning summarizer model: {e}")
                print("This is often due to TensorFlow/Transformers compatibility issues.")
                print("You can skip the summarizer model and still use the other models.")
                import traceback
                traceback.print_exc()
        
        print(f"\nTraining completed! Successfully trained {models_trained} models.")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()