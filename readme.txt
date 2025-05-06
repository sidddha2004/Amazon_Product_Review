Step_1:

Upload the 6 Python files to Colab
Upload the following files:

1_setup.py (Setup and utility functions)
2_preprocessing.py (Text preprocessing)
3_data_processing.py (Data loading and processing)
4_feature_engineering.py (Feature extraction)
5_model.py (Deep learning model)
6_analysis.py (Result analysis)




Run the setup and import modules
python# Run setup
%run 1_setup.py

Step_2:

# Import all required modules
%run 2_preprocessing.py
%run 3_data_processing.py
%run 4_feature_engineering.py
%run 5_model.py
%run 6_analysis.py

Step_3:

Upload your Amazon reviews dataset
pythonfrom google.colab import files
uploaded = files.upload()  # This will prompt for file upload

# Get the filename
filename = list(uploaded.keys())[0]

Step_4:
Process the data
python# Create processor and preprocessor
text_preprocessor = TextPreprocessor()
data_processor = DataProcessor(text_preprocessor)

# Load and process data (adjust sample_size if needed)
df = data_processor.load_data(filename)
df = data_processor.preprocess_data(df, sample_size=10000)  # Optional sampling

# Create train/val/test splits
train_df, val_df, test_df = data_processor.create_train_val_test_split(df)

# Analyze the dataset
data_processor.analyze_dataset(df)

Step_5:
Extract features and train the model
python# Create feature engineer
feature_engineer = FeatureEngineer(max_features=5000)

# Extract features
train_df = feature_engineer.extract_all_features(train_df)
val_df = feature_engineer.extract_all_features(val_df)
test_df = feature_engineer.extract_all_features(test_df)

# Fit TF-IDF
feature_engineer.fit_tfidf(train_df['processed_text'])

# Create and train model
model = DeepSentimentModel(max_words=10000, max_sequence_length=100, embedding_dim=100)

# Train model
history = model.train(
    train_df['processed_text'].values,
    train_df['sentiment'].values,
    val_df['processed_text'].values,
    val_df['sentiment'].values,
    epochs=5,
    batch_size=64
)

# Plot training history
model.plot_training_history()

Step_6:
Evaluate model performance
python# Evaluate on test data
metrics = model.evaluate(
    test_df['processed_text'].values,
    test_df['sentiment'].values
)

# Get predictions
pred_labels, _ = model.predict(test_df['processed_text'].values)
test_df['predicted_sentiment'] = pred_labels

Step_7:
Add rule-based improvements
python# Add rule-based classifier for improved performance
import re

def rule_based_classifier(review):
    """Simple rule-based classifier for negative reviews"""
    review_lower = review.lower()
    
    # Very strong negative indicators
    strong_negative = ["not worth", "waste of money", "terrible", "horrible", 
                       "do not buy", "worst", "useless", "junk", "broke"]
    
    # Moderate negative indicators
    moderate_negative = ["disappointing", "poor", "bad", "doesn't work", 
                        "wouldn't recommend", "overpriced"]
    
    # Positive indicators
    positive = ["great", "excellent", "love", "perfect", "recommend", 
               "amazing", "awesome", "best", "good"]
    
    # Count indicators in each category
    strong_neg_count = sum(1 for phrase in strong_negative if phrase in review_lower)
    moderate_neg_count = sum(1 for phrase in moderate_negative if phrase in review_lower)
    positive_count = sum(1 for phrase in positive if phrase in review_lower)
    
    # Decision logic
    if strong_neg_count > 0:
        return "negative"
    if moderate_neg_count > 0 and positive_count == 0:
        return "negative"
    if positive_count > 0 and (strong_neg_count + moderate_neg_count) == 0:
        return "positive"
    
    # Return neutral if no clear indicators or mixed signals
    return "neutral"

def hybrid_sentiment_classifier(review, model_analyzer):
    """Hybrid sentiment classifier combining model predictions with rules"""
    # Get model prediction
    result = model_analyzer.analyze_review(review)
    model_sentiment = result['sentiment']
    scores = {k: v for k, v in result.items() if k.endswith('_score')}
    
    # Get rule-based prediction
    rule_sentiment = rule_based_classifier(review)
    
    # Logic for combining predictions
    model_confidence = max(scores.values()) if scores else 0.5
    
    # If model is uncertain or rule strongly disagrees with model
    if model_confidence < 0.7 or (rule_sentiment == "negative" and model_sentiment != "negative"):
        return rule_sentiment
    
    # Otherwise trust the model
    return model_sentiment
Step_8:
Create analyzer for interactive testing
python# Create analyzer for testing
analyzer = SentimentAnalyzer(model, text_preprocessor)

# Create interactive analyzer for testing reviews
analyzer.create_interactive_analyzer()

Test with problematic examples
python# Test with some problematic examples
test_reviews = [
    "Very bad quality recieved. Not worth the money.",
    "Not worth the price. Broke within a week.",
    "terrible product that doesn't work properly",
    "waste of money don't buy it",
    "This is a great product, works perfectly",
    "The quality is mediocre but it serves its purpose"
]

# Test with the hybrid approach
for review in test_reviews:
    model_sentiment = analyzer.analyze_review(review)['sentiment']
    hybrid_sentiment = hybrid_sentiment_classifier(review, analyzer)
    
    print(f"\nReview: {review}")
    print(f"Model sentiment: {model_sentiment}")
    print(f"Hybrid sentiment: {hybrid_sentiment}")
