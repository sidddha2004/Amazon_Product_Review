import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class ReviewHelpfulnessModel:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize the model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on the selected type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def extract_features(self, df):
        """
        Extract features relevant for helpfulness prediction
        """
        features = pd.DataFrame()
        
        # Text length features
        features['review_length'] = df['review_text'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        features['title_length'] = df['review_title'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        
        # Readability features
        try:
            import textstat
            features['readability_score'] = df['review_text'].apply(
                lambda x: textstat.flesch_reading_ease(x) if isinstance(x, str) else 0
            )
        except ImportError:
            print("textstat not available, skipping readability features")
        
        # Rating features
        features['rating'] = df['rating']
        features['rating_squared'] = df['rating'] ** 2
        features['is_extreme_rating'] = df['rating'].apply(
            lambda x: 1 if x in [1, 5] else 0
        )
        
        # Sentiment features using NLTK's VADER
        features['sentiment_pos'] = df['review_text'].apply(
            lambda x: self.sia.polarity_scores(x)['pos'] if isinstance(x, str) else 0
        )
        features['sentiment_neg'] = df['review_text'].apply(
            lambda x: self.sia.polarity_scores(x)['neg'] if isinstance(x, str) else 0
        )
        features['sentiment_compound'] = df['review_text'].apply(
            lambda x: self.sia.polarity_scores(x)['compound'] if isinstance(x, str) else 0
        )
        
        # Text structure features
        features['has_question'] = df['review_text'].apply(
            lambda x: 1 if '?' in x else 0 if isinstance(x, str) else 0
        )
        features['sentence_count'] = df['review_text'].apply(
            lambda x: len(nltk.sent_tokenize(x)) if isinstance(x, str) else 0
        )
        features['avg_sentence_length'] = features.apply(
            lambda row: row['review_length'] / max(row['sentence_count'], 1), axis=1
        )
        
        # Language features
        features['capitalized_ratio'] = df['review_text'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1) if isinstance(x, str) else 0
        )
        
        # Categorical features (one-hot encoding)
        if 'category' in df.columns:
            category_dummies = pd.get_dummies(df['category'], prefix='category')
            features = pd.concat([features, category_dummies], axis=1)
        
        if 'brand' in df.columns:
            brand_counts = df['brand'].value_counts()
            top_brands = brand_counts[brand_counts >= 10].index
            df['brand_grouped'] = df['brand'].apply(lambda x: x if x in top_brands else 'other')
            brand_dummies = pd.get_dummies(df['brand_grouped'], prefix='brand')
            features = pd.concat([features, brand_dummies], axis=1)
        
        return features
    
    def create_labels(self, df):
        """
        Create binary labels for helpfulness
        
        This is based on the available data. Different options:
        1. If helpfulVotes exists, use ratio of helpful to total votes
        2. If no votes exist, use review length as a proxy
        """
        if 'reviews.helpfulVotes' in df.columns:
            # Use actual helpful votes if available
            df['helpfulness_ratio'] = df['reviews.helpfulVotes'] / df['reviews.totalVotes'].clip(lower=1)
            labels = (df['helpfulness_ratio'] > 0.5).astype(int)
        elif 'reviews.numHelpful' in df.columns:
            # Alternative helpful votes column name
            labels = (df['reviews.numHelpful'] > 0).astype(int)
        else:
            # Use review length as a proxy for helpfulness
            # Longer, more detailed reviews tend to be more helpful
            median_length = df['review_length'].median()
            labels = (df['review_length'] > median_length).astype(int)
            
        return labels
    
    def train(self, X_train, y_train):
        """
        Train the helpfulness prediction model
        """
        # Scale numerical features
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            print("\nTop 10 important features:")
            print(feature_importance.head(10))
        
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data
        """
        # Scale numerical features
        numeric_features = X_test.select_dtypes(include=['int64', 'float64']).columns
        X_test_scaled = X_test.copy()
        X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
            return y_pred, y_prob
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        # Make predictions
        y_pred = self.predict(X_test)
        if isinstance(y_pred, tuple):
            y_pred, y_prob = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        
        # Return evaluation metrics
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report
        }
    
    def save_model(self, model_path):
        """
        Save the trained model
        """
        import pickle
        with open(model_path, 'wb') as file:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type
            }, file)
        print(f"Helpfulness model saved to {model_path}")
    
    def load_model(self, model_path):
        """
        Load a trained model
        """
        import pickle
        with open(model_path, 'rb') as file:
            saved_data = pickle.load(file)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.model_type = saved_data['model_type']
        print(f"Helpfulness model loaded from {model_path}")