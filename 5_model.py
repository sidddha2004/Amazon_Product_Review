# 5_model.py - Deep learning models for Amazon Review Sentiment Analysis

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import gc

class DeepSentimentModel:
    """Deep learning model for sentiment analysis using LSTM"""
    
    def __init__(self, max_words=10000, max_sequence_length=100, embedding_dim=100):
        """Initialize the model with parameters
        
        Args:
            max_words (int): Maximum number of words in vocabulary
            max_sequence_length (int): Maximum length of input sequences
            embedding_dim (int): Dimensionality of embeddings
        """
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
    
    def preprocess_data(self, texts, labels=None):
        """Preprocess texts and labels for the model
        
        Args:
            texts (array-like): List of text documents
            labels (array-like, optional): List of labels
            
        Returns:
            tuple: (X, y, num_classes) if labels provided, else X
        """
        # Handle empty texts
        texts = [text if isinstance(text, str) else "" for text in texts]
        
        # Fit tokenizer if not fit yet
        if not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
            print("Fitting tokenizer on texts...")
            self.tokenizer.fit_on_texts(texts)
            print(f"Tokenizer vocabulary size: {len(self.tokenizer.word_index)}")
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        if labels is not None:
            # Encode labels if not encoded yet
            if not hasattr(self.label_encoder, 'classes_') or self.label_encoder.classes_.size == 0:
                self.label_encoder.fit(labels)
                print(f"Label encoder classes: {self.label_encoder.classes_}")
            
            y = self.label_encoder.transform(labels)
            
            # Convert to categorical for multi-class
            num_classes = len(self.label_encoder.classes_)
            if num_classes > 2:
                y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
            
            return X, y, num_classes
        
        return X
    
    def build_model(self, num_classes):
        """Build an enhanced Bidirectional LSTM model for sentiment analysis
        
        Args:
            num_classes (int): Number of target classes
            
        Returns:
            tf.keras.Model: Built model
        """
        print(f"Building deep learning model with {num_classes} output classes...")
        
        # Use min to prevent index error when vocab is smaller than max_words
        vocab_size = min(self.max_words, len(self.tokenizer.word_index) + 1)
        
        # Input layer
        inputs = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        x = Embedding(input_dim=vocab_size, 
                      output_dim=self.embedding_dim,
                      input_length=self.max_sequence_length)(inputs)
        
        # Bidirectional LSTM layers
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=True))(x)
        
        # Global pooling
        x = GlobalMaxPooling1D()(x)
        
        # Dense layers with more capacity
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        if num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(num_classes, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Use Adam optimizer with learning rate scheduler
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        if num_classes == 2:
            model.compile(optimizer=optimizer, 
                         loss='binary_crossentropy', 
                         metrics=['accuracy', 
                                 tf.keras.metrics.Precision(),
                                 tf.keras.metrics.Recall()])
        else:
            model.compile(optimizer=optimizer, 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy',
                                 tf.keras.metrics.Precision(),
                                 tf.keras.metrics.Recall()])
        
        self.model = model
        print(f"Model built with {model.count_params():,} parameters")
        return model
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
             epochs=5, batch_size=32, callbacks=None):
        """Train the sentiment analysis model
        
        Args:
            train_texts (array-like): Training text data
            train_labels (array-like): Training labels
            val_texts (array-like, optional): Validation text data
            val_labels (array-like, optional): Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            callbacks (list, optional): List of Keras callbacks
            
        Returns:
            History: Training history
        """
        start_time = time.time()
        print("Preprocessing data for training...")
        
        # Clear previous TF session to avoid memory issues
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Preprocess data
        X_train, y_train, num_classes = self.preprocess_data(train_texts, train_labels)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(num_classes)
        
        # Setup validation data
        validation_data = None
        if val_texts is not None and val_labels is not None:
            X_val, y_val, _ = self.preprocess_data(val_texts, val_labels)
            validation_data = (X_val, y_val)
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if validation_data is not None else 'loss', 
                    patience=3, 
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if validation_data is not None else 'loss', 
                    factor=0.5, 
                    patience=2, 
                    min_lr=0.0001, 
                    verbose=1
                )
            ]
        
        # Train model
        print(f"Starting training for {epochs} epochs with batch size {batch_size}...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self.history = history
        return history
    
    def predict(self, texts):
        """Make predictions on new texts
        
        Args:
            texts (array-like): List of text documents
            
        Returns:
            tuple: (pred_labels, predictions)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess texts
        X = self.preprocess_data(texts)
        
        # Get predictions
        predictions = self.model.predict(X)
        
        # Process predictions based on model type
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class case
            pred_classes = np.argmax(predictions, axis=1)
        else:
            # Binary case
            pred_classes = (predictions > 0.5).astype(int).flatten()
        
        # Convert to original labels
        pred_labels = self.label_encoder.inverse_transform(pred_classes)
        
        return pred_labels, predictions
    
    def evaluate(self, texts, labels):
        """Evaluate model on test data
        
        Args:
            texts (array-like): List of text documents
            labels (array-like): List of true labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess data
        X, y, _ = self.preprocess_data(texts, labels)
        
        # Evaluate model
        results = self.model.evaluate(X, y, verbose=1)
        
        # Format results into a dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        # Get predictions for classification report
        pred_labels, _ = self.predict(texts)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(labels, pred_labels))
        
        # Create confusion matrix
        cm = confusion_matrix(labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png')
        plt.show()
        
        return metrics
    
    def plot_training_history(self):
        """Plot training history
        
        Returns:
            Figure: Matplotlib figure
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet")
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'])
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'])
            ax1.legend(['Train', 'Validation'], loc='lower right')
        else:
            ax1.legend(['Train'], loc='lower right')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        
        # Plot loss
        ax2.plot(self.history.history['loss'])
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'])
            ax2.legend(['Train', 'Validation'], loc='upper right')
        else:
            ax2.legend(['Train'], loc='upper right')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig('visualizations/training_history.png')
        plt.show()
        
        return fig

class SentimentEnsembleModel:
    """Ensemble model combining deep learning and traditional ML models"""
    
    def __init__(self, deep_model=None, rf_model=None, feature_engineer=None):
        """Initialize the ensemble model
        
        Args:
            deep_model (DeepSentimentModel, optional): Deep learning model
            rf_model (RandomForestClassifier, optional): Random forest model
            feature_engineer (FeatureEngineer, optional): Feature engineering component
        """
        self.deep_model = deep_model
        self.rf_model = rf_model
        self.feature_engineer = feature_engineer
    
    def train(self, train_df, val_df, text_column='processed_text', label_column='sentiment'):
        """Train the ensemble model
        
        Args:
            train_df (pd.DataFrame): Training dataframe
            val_df (pd.DataFrame): Validation dataframe
            text_column (str): Column name for text
            label_column (str): Column name for labels
            
        Returns:
            self: For method chaining
        """
        print("===== Training Ensemble Model =====")
        
        # 1. Train the deep learning model
        if self.deep_model is None:
            from tensorflow.keras import backend as K
            K.clear_session()  # Clear session to avoid memory leaks
            self.deep_model = DeepSentimentModel(max_words=10000, 
                                                max_sequence_length=100, 
                                                embedding_dim=100)
        
        print("\n----- Training Deep Learning Model -----")
        self.deep_model.train(
            train_df[text_column].values,
            train_df[label_column].values,
            val_df[text_column].values if not val_df.empty else None,
            val_df[label_column].values if not val_df.empty else None,
            epochs=5,
            batch_size=64
        )
        
        # 2. Train the random forest model with TF-IDF features
        print("\n----- Training Random Forest Model -----")
        
        # Import feature engineer if needed
        if self.feature_engineer is None:
            # We need to have the FeatureEngineer class available locally
            # This would be handled by imports, but for Colab compatibility:
            self.feature_engineer = FeatureEngineer(max_features=5000)
            self.feature_engineer.fit_tfidf(train_df[text_column].values)
        
        X_train_tfidf = self.feature_engineer.transform_tfidf(train_df[text_column].values)
        
        if self.rf_model is None:
            self.rf_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=20, 
                n_jobs=-1, 
                random_state=42
            )
        
        self.rf_model.fit(X_train_tfidf, train_df[label_column].values)
        print("Random Forest model trained")
        
        # 3. Evaluate individual models
        print("\n===== Evaluating Models on Validation Set =====")
        
        if not val_df.empty:
            # Deep learning model
            dl_metrics = self.deep_model.evaluate(
                val_df[text_column].values,
                val_df[label_column].values
            )
            print(f"Deep Learning Model Metrics: {dl_metrics}")
            
            # Random forest model
            X_val_tfidf = self.feature_engineer.transform_tfidf(val_df[text_column].values)
            rf_preds = self.rf_model.predict(X_val_tfidf)
            rf_accuracy = np.mean(rf_preds == val_df[label_column].values)
            print(f"Random Forest Model Accuracy: {rf_accuracy:.4f}")
        else:
            print("No validation data available for evaluation.")
        
        return self
    
    def predict(self, texts, weights=None):
        """Make ensemble predictions using weighted voting
        
        Args:
            texts (array-like): List of text documents
            weights (list, optional): Model weights [deep_model_weight, rf_model_weight]
            
        Returns:
            tuple: (ensemble_labels, ensemble_probs)
        """
        if self.deep_model is None or self.rf_model is None:
            raise ValueError("Models have not been trained yet")
        
        # Default weights if not provided
        if weights is None:
            weights = [0.7, 0.3]  # Deep learning model has higher weight by default
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        dl_weight, rf_weight = weights
        
        # Get deep learning predictions
        dl_labels, dl_probs = self.deep_model.predict(texts)
        
        # Get random forest predictions
        X_tfidf = self.feature_engineer.transform_tfidf(texts)
        rf_labels = self.rf_model.predict(X_tfidf)
        rf_probs = self.rf_model.predict_proba(X_tfidf)
        
        # Get the label encoder classes
        classes = self.deep_model.label_encoder.classes_
        num_classes = len(classes)
        
        # For binary classification
        if num_classes == 2:
            # Combine probabilities
            if len(dl_probs.shape) == 1 or dl_probs.shape[1] == 1:
                dl_pos_probs = dl_probs.flatten()
                dl_neg_probs = 1 - dl_pos_probs
                dl_probs_mat = np.column_stack((dl_neg_probs, dl_pos_probs))
            else:
                dl_probs_mat = dl_probs
                
            ensemble_probs = dl_weight * dl_probs_mat + rf_weight * rf_probs
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            
        # For multi-class
        else:
            # Create weighted probability matrix
            ensemble_probs = dl_weight * dl_probs + rf_weight * rf_probs
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        # Convert numeric predictions back to labels
        ensemble_labels = self.deep_model.label_encoder.inverse_transform(ensemble_preds)
        
        return ensemble_labels, ensemble_probs
    
    def evaluate(self, texts, true_labels):
        """Evaluate the ensemble model
        
        Args:
            texts (array-like): List of text documents
            true_labels (array-like): List of true labels
            
        Returns:
            tuple: (accuracy, classification_report)
        """
        # Get ensemble predictions
        pred_labels, _ = self.predict(texts)
        
        # Calculate accuracy
        accuracy = np.mean(pred_labels == true_labels)
        
        # Generate classification report
        report = classification_report(true_labels, pred_labels)
        
        print("\n===== Ensemble Model Evaluation =====")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.deep_model.label_encoder.classes_,
                   yticklabels=self.deep_model.label_encoder.classes_)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Ensemble Model Confusion Matrix')
        plt.tight_layout()
        plt.savefig('visualizations/ensemble_confusion_matrix.png')
        plt.show()
        
        return accuracy, report

# Test function
def test_models():
    """Test model functionality with sample data"""
    # Create sample data
    texts = [
        "This product is amazing! I love it so much.",
        "Terrible quality, broke after one day. Would not recommend!",
        "It's okay, not great but not bad either. Average product."
    ]
    
    labels = ["positive", "negative", "neutral"]
    
    # Test deep learning model
    deep_model = DeepSentimentModel(max_words=100, max_sequence_length=20, embedding_dim=50)
    X, y, num_classes = deep_model.preprocess_data(texts, labels)
    
    # Build and train with minimal settings for testing
    deep_model.build_model(num_classes)
    history = deep_model.train(texts, labels, epochs=2, batch_size=1)
    
    # Test prediction
    pred_labels, pred_probs = deep_model.predict(texts)
    print(f"Predicted labels: {pred_labels}")
    
    print("Model testing completed!")
    return deep_model

if __name__ == "__main__":
    test_models()