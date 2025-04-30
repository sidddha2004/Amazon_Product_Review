import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import spacy
from collections import Counter
import os
class AspectBasedSentimentModel:
    def __init__(self, max_words=10000, max_sequence_length=200, embedding_dim=100):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.text_tokenizer = Tokenizer(num_words=max_words)
        self.aspect_tokenizer = Tokenizer(num_words=1000)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.nlp = spacy.load('en_core_web_sm')
        self.aspect_categories = None
    
    def extract_aspects(self, texts, num_aspects=20, min_count=5):
        """
        Extract common aspects from review texts
        """
        print("Extracting aspect terms from reviews...")
        
        # Extract noun phrases as potential aspects
        aspect_candidates = []
        
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                continue
            
            doc = self.nlp(text)
            
            # Extract nouns and noun phrases
            for chunk in doc.noun_chunks:
                if 1 <= len(chunk.text.split()) <= 3:  # Limit to short phrases
                    aspect_candidates.append(chunk.text.lower())
            
            # Also include important product nouns
            for token in doc:
                if token.pos_ == 'NOUN' and not token.is_stop:
                    aspect_candidates.append(token.text.lower())
        
        # Count aspect occurrences
        aspect_counts = Counter(aspect_candidates)
        
        # Filter aspects that appear frequently enough
        common_aspects = [aspect for aspect, count in aspect_counts.most_common(100) 
                        if count >= min_count][:num_aspects]
        
        print(f"Extracted {len(common_aspects)} common aspects: {common_aspects[:10]}...")
        
        self.aspect_categories = common_aspects
        return common_aspects
    
    def create_aspect_training_data(self, texts, labels, aspects=None):
        """
        Create training data pairs (text, aspect) with corresponding sentiment labels
        """
        if aspects is None:
            aspects = self.aspect_categories
            
        if aspects is None:
            aspects = self.extract_aspects(texts)
        
        # For each review, find mentioned aspects and create training examples
        training_texts = []
        training_aspects = []
        training_labels = []
        
        for text, label in zip(texts, labels):
            if not isinstance(text, str) or not text.strip():
                continue
                
            # Find aspects mentioned in this review
            doc = self.nlp(text.lower())
            text_tokens = set(token.text.lower() for token in doc)
            
            mentioned_aspects = []
            for aspect in aspects:
                aspect_tokens = set(aspect.lower().split())
                # Check if all words in the aspect appear in the text
                if any(aspect_token in text_tokens for aspect_token in aspect_tokens):
                    mentioned_aspects.append(aspect)
            
            # If no aspects found, use the most likely one
            if not mentioned_aspects:
                # Use a general aspect (e.g., "product" or "item")
                mentioned_aspects = ["product"]
            
            # Create a training example for each mentioned aspect
            for aspect in mentioned_aspects:
                training_texts.append(text)
                training_aspects.append(aspect)
                training_labels.append(label)
        
        print(f"Created {len(training_texts)} text-aspect-sentiment triplets for training")
        return training_texts, training_aspects, training_labels
    
    def preprocess_data(self, texts, aspects, labels=None):
        """
        Preprocess data for aspect-based sentiment analysis
        """
        # Fit tokenizers
        self.text_tokenizer.fit_on_texts(texts)
        self.aspect_tokenizer.fit_on_texts(aspects)
        
        # Convert texts and aspects to sequences
        text_sequences = self.text_tokenizer.texts_to_sequences(texts)
        aspect_sequences = self.aspect_tokenizer.texts_to_sequences(aspects)
        
        # Pad sequences
        padded_text_sequences = pad_sequences(text_sequences, maxlen=self.max_sequence_length)
        padded_aspect_sequences = pad_sequences(aspect_sequences, maxlen=20)  # Shorter max length for aspects
        
        # Prepare inputs
        X = [padded_text_sequences, padded_aspect_sequences]
        
        # Prepare outputs if labels are provided
        if labels is not None:
            # Encode labels
            if not hasattr(self.label_encoder, 'classes_') or not self.label_encoder.classes_.size:
                self.label_encoder.fit(labels)
            
            y = self.label_encoder.transform(labels)
            
            # Convert to categorical for multi-class
            num_classes = len(self.label_encoder.classes_)
            if num_classes > 2:
                y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
                
            return X, y
        
        return X
    
    def build_model(self, num_classes=3):
        """
        Build the aspect-based sentiment analysis model
        """
        # Text input
        text_input = Input(shape=(self.max_sequence_length,), name='text_input')
        text_embedding = Embedding(input_dim=self.max_words,
                                  output_dim=self.embedding_dim,
                                  input_length=self.max_sequence_length)(text_input)
        text_lstm = Bidirectional(LSTM(64, return_sequences=False))(text_embedding)
        
        # Aspect input
        aspect_input = Input(shape=(20,), name='aspect_input')  # Shorter length for aspects
        aspect_embedding = Embedding(input_dim=1000,  # Smaller vocabulary for aspects
                                    output_dim=50,    # Smaller embedding for aspects
                                    input_length=20)(aspect_input)
        aspect_lstm = Bidirectional(LSTM(32, return_sequences=False))(aspect_embedding)
        
        # Combine text and aspect features
        combined = Concatenate()([text_lstm, aspect_lstm])
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(combined)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        
        # Output layer
        if num_classes == 2:
            output_layer = Dense(1, activation='sigmoid')(dropout2)
        else:
            output_layer = Dense(num_classes, activation='softmax')(dropout2)
        
        # Create model
        model = Model(inputs=[text_input, aspect_input], outputs=output_layer)
        
        # Compile model
        if num_classes == 2:
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=32):
        """
        Train the aspect-based sentiment model
        """
        # Build model if not already built
        if self.model is None:
            num_classes = 2 if len(y_train.shape) == 1 else y_train.shape[1]
            self.build_model(num_classes)
        
        # Set up validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss' if validation_data else 'loss', 
                                           patience=2, 
                                           restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', 
                                               factor=0.5, 
                                               patience=1)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, texts, aspects=None):
        """
        Make predictions on text-aspect pairs
        """
        # If aspects are not provided, use the first aspect for each text
        if aspects is None:
            aspects = [self.aspect_categories[0]] * len(texts)
        
        # Preprocess inputs
        X = self.preprocess_data(texts, aspects)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Convert to class labels
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Multi-class case
            predicted_classes = np.argmax(predictions, axis=1)
        else:
            # Binary case
            predicted_classes = (predictions > 0.5).astype('int').flatten()
        
        # Convert to original labels
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        return predicted_labels, predictions
    
    # In aspect_model.py and sentiment_model.py, find the save_model method
    def save_model(self, model_path, tokenizer_path):
        """
        Save model and tokenizer
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Create a SavedModel directory instead of .keras file
        saved_model_dir = model_path
        self.model.save(saved_model_dir, save_format='tf')
        
        # Save tokenizer and label encoder
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump({
                'tokenizer': self.tokenizer, 
                'label_encoder': self.label_encoder,
                'max_words': self.max_words,
                'max_sequence_length': self.max_sequence_length,
                'embedding_dim': self.embedding_dim
            }, handle)
        
        print(f"Model saved to {saved_model_dir}")
        print(f"Tokenizer saved to {tokenizer_path}")

    # And update the load_model method
    def load_model(self, model_path, tokenizer_path):
        """
        Load model and tokenizer
        """
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load tokenizer and label encoder
        with open(tokenizer_path, 'rb') as handle:
            saved_data = pickle.load(handle)
            self.tokenizer = saved_data['tokenizer']
            self.label_encoder = saved_data['label_encoder']
            
            # Load parameters if available
            if 'max_words' in saved_data:
                self.max_words = saved_data['max_words']
            if 'max_sequence_length' in saved_data:
                self.max_sequence_length = saved_data['max_sequence_length']
            if 'embedding_dim' in saved_data:
                self.embedding_dim = saved_data['embedding_dim']
        
        print(f"Model loaded from {model_path}")
        print(f"Tokenizer loaded from {tokenizer_path}")