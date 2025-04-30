import numpy as np
import tensorflow as tf
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize

class ReviewSummarizerModel:
    def __init__(self, model_name='t5-small', max_input_length=512, max_output_length=100):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = TFT5ForConditionalGeneration.from_pretrained(model_name)
    
    def extractive_summarize(self, text, num_sentences=2):
        """
        Create an extractive summary by selecting important sentences
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
            
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # If text is short enough, return as is
        if len(sentences) <= num_sentences:
            return text
        
        # Score sentences based on position and length
        scores = []
        for i, sentence in enumerate(sentences):
            # Position score (first and last sentences are important)
            position_score = 1.0
            if i == 0:
                position_score = 2.0  # First sentence
            elif i == len(sentences) - 1:
                position_score = 1.5  # Last sentence
            
            # Length score (prefer medium-length sentences)
            words = len(sentence.split())
            if 5 <= words <= 20:
                length_score = 1.0
            elif words < 5:
                length_score = 0.5  # Too short
            else:
                length_score = 0.8  # Long sentence
            
            scores.append(position_score * length_score)
        
        # Select top sentences
        top_indices = np.argsort(scores)[-num_sentences:]
        top_indices = sorted(top_indices)  # Sort to maintain original order
        
        # Create summary
        summary = ' '.join([sentences[i] for i in top_indices])
        
        return summary
    
    def abstractive_summarize(self, text):
        """
        Generate an abstractive summary using T5 model
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
            
        # Prepare input for T5
        input_text = "summarize: " + text
        
        # Tokenize
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="tf",
            max_length=self.max_input_length,
            truncation=True
        )
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs,
            max_length=self.max_output_length,
            min_length=10,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def fine_tune(self, train_texts, train_summaries, validation_texts=None, validation_summaries=None, epochs=3, batch_size=4):
        """
        Fine-tune the T5 model on review summarization
        """
        print("Fine-tuning T5 model for review summarization...")
        
        # Prepare training data
        train_inputs = ["summarize: " + text for text in train_texts]
        train_targets = train_summaries
        
        # Tokenize inputs
        train_encodings = self.tokenizer(
            train_inputs,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors="tf"
        )
        
        # Tokenize targets
        target_encodings = self.tokenizer(
            train_targets,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors="tf"
        )
        
        # Convert to TF Dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask']
            },
            target_encodings['input_ids']
        ))
        
        # Prepare validation data if provided
        validation_dataset = None
        if validation_texts is not None and validation_summaries is not None:
            val_inputs = ["summarize: " + text for text in validation_texts]
            
            val_encodings = self.tokenizer(
                val_inputs,
                max_length=self.max_input_length,
                padding='max_length',
                truncation=True,
                return_tensors="tf"
            )
            
            val_target_encodings = self.tokenizer(
                validation_summaries,
                max_length=self.max_output_length,
                padding='max_length',
                truncation=True,
                return_tensors="tf"
            )
            
            validation_dataset = tf.data.Dataset.from_tensor_slices((
                {
                    'input_ids': val_encodings['input_ids'],
                    'attention_mask': val_encodings['attention_mask']
                },
                val_target_encodings['input_ids']
            ))
            
            validation_dataset = validation_dataset.batch(batch_size)
        
        # Prepare training dataset
        train_dataset = train_dataset.batch(batch_size)
        
        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        
        self.model.compile(
            optimizer=optimizer,
            loss=self._compute_loss
        )
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=validation_dataset
        )
        
        return history
    
    def _compute_loss(self, labels, logits):
        """
        Compute loss function for T5 model
        """
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        
        # Create mask to ignore padded tokens
        mask = tf.cast(labels != 0, dtype=tf.float32)
        
        # Compute loss
        loss = loss_fn(labels, logits) * mask
        
        # Compute mean loss over non-padded tokens
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    def save_model(self, save_dir):
        """
        Save the T5 model and tokenizer
        """
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Saved summarizer model to {save_dir}")
    
    def load_model(self, load_dir):
        """
        Load a saved T5 model and tokenizer
        """
        self.model = TFT5ForConditionalGeneration.from_pretrained(load_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(load_dir)
        print(f"Loaded summarizer model from {load_dir}")