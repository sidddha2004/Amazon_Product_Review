# scripts/evaluation.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.amazon_processor import AmazonDataProcessor
from src.models.sentiment_model import SentimentAnalysisModel
from src.models.aspect_model import AspectBasedSentimentModel
from src.models.helpfulness_model import ReviewHelpfulnessModel

def evaluate_sentiment_model(model, test_df, output_dir):
    """
    Comprehensive evaluation of the sentiment analysis model
    """
    print("\nEvaluating Sentiment Analysis Model...")
    
    # Determine text column
    text_column = 'cleaned_text' if 'cleaned_text' in test_df.columns else 'review_text'
    
    # Get predictions
    predicted_labels, predicted_scores = model.predict(test_df[text_column].values)
    true_labels = test_df['sentiment'].values
    
    # Convert labels to numeric values
    true_numeric = model.label_encoder.transform(true_labels)
    pred_numeric = model.label_encoder.transform(predicted_labels)
    
    # Calculate basic metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=model.label_encoder.classes_,
               yticklabels=model.label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Sentiment Analysis')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sentiment_confusion_matrix.png'))
    plt.close()
    
    # ROC Curve (for each class in one-vs-rest fashion)
    plt.figure(figsize=(10, 8))
    
    if predicted_scores.shape[1] > 1:  # Multi-class case
        for i, class_name in enumerate(model.label_encoder.classes_):
            # One-vs-Rest approach
            true_binary = (true_numeric == i).astype(int)
            
            # Get scores for this class
            if len(predicted_scores.shape) > 1:
                class_scores = predicted_scores[:, i]
            else:
                class_scores = predicted_scores if i == 1 else 1 - predicted_scores
                
            # Calculate ROC
            fpr, tpr, _ = roc_curve(true_binary, class_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    else:  # Binary case
        # If shape is (n,) - binary prediction
        fpr, tpr, _ = roc_curve(true_numeric, predicted_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'sentiment_roc_curve.png'))
    plt.close()
    
    # Additional metrics
    metrics = {
        'accuracy': accuracy,
        'class_metrics': classification_report(true_labels, predicted_labels, output_dict=True),
        'confusion_matrix': cm.tolist(),
    }
    
    return metrics

def evaluate_helpfulness_model(model, test_df, output_dir):
    """
    Comprehensive evaluation of the helpfulness prediction model
    """
    print("\nEvaluating Helpfulness Model...")
    
    # Prepare features
    features = model.extract_features(test_df)
    
    # Get true labels
    true_labels = model.create_labels(test_df)
    
    # Get predictions
    predictions = model.predict(features)
    if isinstance(predictions, tuple):
        pred_labels, pred_probs = predictions
    else:
        pred_labels = predictions
        pred_probs = None
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Not Helpful', 'Helpful'],
               yticklabels=['Not Helpful', 'Helpful'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Review Helpfulness')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'helpfulness_confusion_matrix.png'))
    plt.close()
    
    # ROC Curve (if probabilities are available)
    if pred_probs is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Helpfulness Prediction')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'helpfulness_roc_curve.png'))
        plt.close()
        
        # Precision-Recall curve
        plt.figure(figsize=(8, 6))
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, pred_probs)
        plt.plot(recall_curve, precision_curve, label=f'Precision-Recall')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Helpfulness Prediction')
        plt.savefig(os.path.join(output_dir, 'helpfulness_precision_recall.png'))
        plt.close()
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
    }
    
    return metrics

def main():
    """
    Main function to run the evaluation
    """
    parser = argparse.ArgumentParser(description='Evaluate Amazon review analysis models')
    parser.add_argument('--data_file', type=str, required=True, help='Path to processed data file')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default='evaluation', help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Load test data
    data_processor = AmazonDataProcessor()
    df = data_processor.load_data(args.data_file)
    
    if df is None:
        print(f"Error: Could not load data from {args.data_file}")
        return
    
    # Process data
    clean_df = data_processor.clean_dataset(df)
    model_df = data_processor.prepare_model_data(clean_df)
    
    # Split data (we only need the test set)
    _, _, test_df = data_processor.split_data(model_df)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and evaluate sentiment model
    sentiment_model_path = os.path.join(args.model_dir, 'sentiment_model')
    sentiment_tokenizer_path = os.path.join(args.model_dir, 'sentiment_tokenizer.pkl')
    
    if os.path.exists(sentiment_model_path) and os.path.exists(sentiment_tokenizer_path):
        print("\nEvaluating Sentiment Analysis Model...")
        sentiment_model = SentimentAnalysisModel()
        try:
            sentiment_model.load_model(sentiment_model_path, sentiment_tokenizer_path)
            sentiment_metrics = evaluate_sentiment_model(sentiment_model, test_df, args.output_dir)
            
            # Save metrics to file
            with open(os.path.join(args.output_dir, 'sentiment_metrics.txt'), 'w') as f:
                for key, value in sentiment_metrics.items():
                    if key != 'confusion_matrix' and key != 'class_metrics':
                        f.write(f"{key}: {value}\n")
                f.write("\nDetailed classification report:\n")
                f.write(classification_report(test_df['sentiment'], 
                                             sentiment_model.predict(test_df['cleaned_text'])[0]))
        except Exception as e:
            print(f"Error evaluating sentiment model: {e}")
    
    # Load and evaluate helpfulness model
    helpfulness_model_path = os.path.join(args.model_dir, 'helpfulness_model.pkl')
    
    if os.path.exists(helpfulness_model_path):
        print("\nEvaluating Helpfulness Model...")
        helpfulness_model = ReviewHelpfulnessModel()
        try:
            helpfulness_model.load_model(helpfulness_model_path)
            helpfulness_metrics = evaluate_helpfulness_model(helpfulness_model, test_df, args.output_dir)
            
            # Save metrics to file
            with open(os.path.join(args.output_dir, 'helpfulness_metrics.txt'), 'w') as f:
                for key, value in helpfulness_metrics.items():
                    if key != 'confusion_matrix':
                        f.write(f"{key}: {value}\n")
        except Exception as e:
            print(f"Error evaluating helpfulness model: {e}")
    
    print("\nEvaluation completed. Results saved to:", args.output_dir)

if __name__ == '__main__':
    main()