import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input, Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class SentimentAnalysisModel:
    def __init__(self, max_words=10000, max_sequence_length=200, embedding_dim=100):
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.label_encoder = LabelEncoder()
        self.model = None
    
    def preprocess_data(self, train_texts, train_labels, test_texts=None, test_labels=None):
        """
        Preprocess text data for sentiment analysis
        
        Returns:
        - If test_texts and test_labels are provided: X_train, y_train, X_test, y_test
        - Otherwise: X_train, y_train
        """
        print(f"Preprocessing {len(train_texts)} training texts")
        
        # Convert empty texts to empty strings
        train_texts = [text if isinstance(text, str) else "" for text in train_texts]
        
        # Fit tokenizer on training data
        self.tokenizer.fit_on_texts(train_texts)
        
        # Convert texts to sequences
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        
        # Pad sequences
        X_train = pad_sequences(train_sequences, maxlen=self.max_sequence_length)
        
        # Encode labels
        self.label_encoder.fit(train_labels)
        y_train = self.label_encoder.transform(train_labels)
        
        # Convert to categorical if more than 2 classes
        num_classes = len(self.label_encoder.classes_)
        print(f"Found {num_classes} classes: {self.label_encoder.classes_}")
        
        if num_classes > 2:
            y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
        
        # Process test data if provided
        if test_texts is not None and test_labels is not None and len(test_texts) > 0 and len(test_labels) > 0:
            # Handle empty texts
            test_texts = [text if isinstance(text, str) else "" for text in test_texts]
            
            test_sequences = self.tokenizer.texts_to_sequences(test_texts)
            X_test = pad_sequences(test_sequences, maxlen=self.max_sequence_length)
            y_test = self.label_encoder.transform(test_labels)
            
            if num_classes > 2:
                y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
                
            return X_train, y_train, X_test, y_test
        
        return X_train, y_train
    
    def build_cnn_lstm_model(self, num_classes):
        """
        Build a hybrid CNN-LSTM model for sentiment analysis
        """
        # Input layer
        input_layer = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding_layer = Embedding(
            input_dim=self.max_words,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length
        )(input_layer)
        
        # CNN layers with different filter sizes for feature extraction
        conv1 = Conv1D(filters=128, kernel_size=3, padding='valid', activation='relu')(embedding_layer)
        pool1 = GlobalMaxPooling1D()(conv1)
        
        conv2 = Conv1D(filters=128, kernel_size=4, padding='valid', activation='relu')(embedding_layer)
        pool2 = GlobalMaxPooling1D()(conv2)
        
        conv3 = Conv1D(filters=128, kernel_size=5, padding='valid', activation='relu')(embedding_layer)
        pool3 = GlobalMaxPooling1D()(conv3)
        
        # BiLSTM layer
        lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(embedding_layer)
        lstm_pool = GlobalMaxPooling1D()(lstm_layer)
        
        # Concatenate CNN and LSTM features
        concat = Concatenate()([pool1, pool2, pool3, lstm_pool])
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.5)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.3)(dense2)
        
        # Output layer
        if num_classes == 2:
            output_layer = Dense(1, activation='sigmoid')(dropout2)
        else:
            output_layer = Dense(num_classes, activation='softmax')(dropout2)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        if num_classes == 2:
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        else:
            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam', 
                metrics=['accuracy']
            )
        
        self.model = model
        return model
    
    def build_simple_lstm_model(self, num_classes):
        """
        Build a simpler LSTM model for sentiment analysis
        """
        model = Sequential()
        model.add(Embedding(
            input_dim=self.max_words, 
            output_dim=self.embedding_dim, 
            input_length=self.max_sequence_length
        ))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
        else:
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=64, model_type='cnn_lstm'):
        """
        Train the sentiment analysis model
        """
        # Build model if not already built
        if self.model is None:
            num_classes = 2
            if len(y_train.shape) > 1:
                # Multi-class case
                num_classes = y_train.shape[1]
            else:
                # Binary case
                num_classes = 2
            
            if model_type == 'cnn_lstm':
                self.build_cnn_lstm_model(num_classes)
            else:
                self.build_simple_lstm_model(num_classes)
        
        # Set up validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss', 
                patience=2, 
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss', 
                factor=0.5, 
                patience=1, 
                min_lr=0.0001
            )
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
    
    def predict(self, texts):
        """
        Make predictions on new texts
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        # Handle empty texts
        texts = [text if isinstance(text, str) else "" for text in texts]
            
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Pad sequences
        X = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        # Get predictions
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