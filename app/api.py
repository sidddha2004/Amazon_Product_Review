import os
import sys
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.preprocessor import TextPreprocessor
from src.models.sentiment_model import SentimentAnalysisModel
from src.models.aspect_model import AspectBasedSentimentModel
from src.models.helpfulness_model import ReviewHelpfulnessModel
from src.models.summarizer_model import ReviewSummarizerModel

app = Flask(__name__)

# Global variables for loaded models
models = {}
text_preprocessor = TextPreprocessor()

def load_models(model_dir):
    """
    Load all available models
    """
    global models
    
    # Load sentiment model if available
    sentiment_model_path = os.path.join(model_dir, 'sentiment_model')
    sentiment_tokenizer_path = os.path.join(model_dir, 'sentiment_tokenizer.pkl')
    
    if os.path.exists(sentiment_model_path) and os.path.exists(sentiment_tokenizer_path):
        print("Loading sentiment analysis model...")
        sentiment_model = SentimentAnalysisModel()
        sentiment_model.load_model(sentiment_model_path, sentiment_tokenizer_path)
        models['sentiment'] = sentiment_model
    
    # Load aspect model if available
    aspect_model_path = os.path.join(model_dir, 'aspect_model')
    aspect_tokenizer_path = os.path.join(model_dir, 'aspect_tokenizer.pkl')
    
    if os.path.exists(aspect_model_path) and os.path.exists(aspect_tokenizer_path):
        print("Loading aspect-based sentiment model...")
        aspect_model = AspectBasedSentimentModel()
        aspect_model.load_model(aspect_model_path, aspect_tokenizer_path)
        models['aspect'] = aspect_model
    
    # Load helpfulness model if available
    helpfulness_model_path = os.path.join(model_dir, 'helpfulness_model.pkl')
    
    if os.path.exists(helpfulness_model_path):
        print("Loading review helpfulness model...")
        helpfulness_model = ReviewHelpfulnessModel()
        helpfulness_model.load_model(helpfulness_model_path)
        models['helpfulness'] = helpfulness_model
    
    # Load summarizer model if available
    summarizer_model_path = os.path.join(model_dir, 'summarizer_model')
    
    if os.path.exists(summarizer_model_path):
        print("Loading review summarizer model...")
        summarizer_model = ReviewSummarizerModel()
        summarizer_model.load_model(summarizer_model_path)
        models['summarizer'] = summarizer_model
    
    print(f"Loaded {len(models)} models")

@app.route('/')
def home():
    """Home page with simple form"""
    return render_template('index.html')

@app.route('/api/analyze_review', methods=['POST'])
def analyze_review():
    """
    Analyze a single review
    """
    try:
        data = request.json
        
        if not data or 'review' not in data:
            return jsonify({'error': 'Missing review text'}), 400
        
        review_text = data['review']
        
        # Clean text if requested
        if data.get('preprocess', True):
            processed_text = text_preprocessor.clean_text(review_text)
        else:
            processed_text = review_text
        
        result = {
            'review': review_text,
            'processed_review': processed_text,
            'analysis': {}
        }
        
        # Sentiment analysis
        if 'sentiment' in models:
            sentiment_label, sentiment_scores = models['sentiment'].predict([processed_text])
            
            result['analysis']['sentiment'] = {
                'label': sentiment_label[0]
            }
            
            # Add scores (convert to Python types for JSON serialization)
            if hasattr(sentiment_scores, 'shape') and len(sentiment_scores.shape) > 1:
                # Multi-class case
                result['analysis']['sentiment']['scores'] = {
                    label: float(sentiment_scores[0][i])
                    for i, label in enumerate(models['sentiment'].label_encoder.classes_)
                }
            else:
                # Binary case (assume positive/negative)
                pos_score = float(sentiment_scores[0])
                result['analysis']['sentiment']['scores'] = {
                    'positive': pos_score,
                    'negative': 1.0 - pos_score
                }
        
        # Aspect-based sentiment analysis
        if 'aspect' in models:
            aspects = {}
            
            for aspect in models['aspect'].aspect_categories:
                aspect_labels, aspect_scores = models['aspect'].predict([processed_text], [aspect])
                
                aspects[aspect] = {
                    'sentiment': aspect_labels[0]
                }
                
                # Add scores (convert to Python types for JSON serialization)
                if hasattr(aspect_scores, 'shape') and len(aspect_scores.shape) > 1:
                    # Multi-class case
                    aspects[aspect]['scores'] = {
                        label: float(aspect_scores[0][i])
                        for i, label in enumerate(models['aspect'].label_encoder.classes_)
                    }
                else:
                    # Binary case
                    pos_score = float(aspect_scores[0])
                    aspects[aspect]['scores'] = {
                        'positive': pos_score,
                        'negative': 1.0 - pos_score
                    }
            
            result['analysis']['aspects'] = aspects
        
        # Helpfulness prediction
        if 'helpfulness' in models:
            # Create a DataFrame with the review for feature extraction
            review_df = pd.DataFrame({
                'review_text': [review_text],
                'review_title': [''],
                'rating': [0]  # Default value
            })
            
            # Extract features
            features = models['helpfulness'].extract_features(review_df)
            
            # Make prediction
            helpfulness_pred = models['helpfulness'].predict(features)
            
            if isinstance(helpfulness_pred, tuple):
                helpfulness_label, helpfulness_prob = helpfulness_pred
                result['analysis']['helpfulness'] = {
                    'is_helpful': bool(helpfulness_label[0]),
                    'probability': float(helpfulness_prob[0])
                }
            else:
                result['analysis']['helpfulness'] = {
                    'is_helpful': bool(helpfulness_pred[0])
                }
        
        # Review summarization
        if 'summarizer' in models:
            # Use extractive summarization for short reviews
            if len(review_text.split()) < 100:
                summary = models['summarizer'].extractive_summarize(review_text, num_sentences=1)
            else:
                summary = models['summarizer'].abstractive_summarize(review_text)
                
            result['analysis']['summary'] = summary
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """
    Get information about loaded models
    """
    model_info = {}
    
    if 'sentiment' in models:
        model_info['sentiment'] = {
            'classes': models['sentiment'].label_encoder.classes_.tolist()
        }
    
    if 'aspect' in models:
        model_info['aspect'] = {
            'aspects': models['aspect'].aspect_categories,
            'classes': models['aspect'].label_encoder.classes_.tolist()
        }
    
    if 'helpfulness' in models:
        model_info['helpfulness'] = {
            'model_type': models['helpfulness'].model_type
        }
    
    if 'summarizer' in models:
        model_info['summarizer'] = {
            'model_name': models['summarizer'].model_name
        }
    
    return jsonify(model_info)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Start the Amazon Review Analysis API')
    parser.add_argument('--model_dir', type=str, default='../models', help='Directory with trained models')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the API on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the API on')
    
    args = parser.parse_args()
    
    # Make sure templates directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    
    # Load models
    load_models(args.model_dir)
    
    # Start API
    app.run(host=args.host, port=args.port, debug=True)