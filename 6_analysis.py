# 6_analysis.py - Visualization and analysis for Amazon Review Sentiment Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from wordcloud import WordCloud
import re
from collections import Counter
from IPython.display import display, HTML
from sklearn.preprocessing import LabelEncoder

class SentimentAnalyzer:
    """Class for analyzing and visualizing sentiment analysis results"""
    
    def __init__(self, model, text_preprocessor=None):
        """Initialize with a trained model
        
        Args:
            model: Trained sentiment model (Deep or Ensemble)
            text_preprocessor: Text preprocessor for new reviews
        """
        self.model = model
        self.text_preprocessor = text_preprocessor
    
    def analyze_review(self, review_text):
        """Analyze a single review
        
        Args:
            review_text (str): Review text to analyze
            
        Returns:
            dict: Analysis results
        """
        # Preprocess the text if preprocessor available
        if self.text_preprocessor:
            processed = self.text_preprocessor.preprocess_text(review_text)
        else:
            processed = review_text
        
        # Predict sentiment
        sentiment, probs = self.model.predict([processed])
        
        # Format results
        result = {
            'review': review_text,
            'sentiment': sentiment[0],
        }
        
        # Add probability scores based on model type
        if hasattr(self.model, 'deep_model'):
            # Ensemble model
            classes = self.model.deep_model.label_encoder.classes_
        elif hasattr(self.model, 'label_encoder'):
            # Deep model
            classes = self.model.label_encoder.classes_
        else:
            # Fallback for test models
            if hasattr(self.model, 'classes_'):
                classes = self.model.classes_
            else:
                # Default classes for testing
                classes = np.array(['negative', 'neutral', 'positive'])
            
        # Format probabilities based on shape
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            # Multi-class case
            result['confidence'] = float(np.max(probs[0]))
            for i, class_name in enumerate(classes):
                result[f'{class_name}_score'] = float(probs[0][i])
        else:
            # Binary case
            if len(probs.shape) == 1:
                pos_prob = float(probs[0])
            else:
                pos_prob = float(probs[0][0])
                
            result['positive_score'] = pos_prob
            result['negative_score'] = 1 - pos_prob
            result['confidence'] = max(pos_prob, 1 - pos_prob)
        
        return result
    
    def analyze_batch(self, reviews, batch_size=32):
        """Analyze a batch of reviews
        
        Args:
            reviews (list): List of review texts
            batch_size (int): Batch size for processing
            
        Returns:
            list: Analysis results for each review
        """
        results = []
        
        # Process in batches to save memory
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            
            # Preprocess if preprocessor available
            if self.text_preprocessor:
                processed_batch = [self.text_preprocessor.preprocess_text(review) for review in batch]
            else:
                processed_batch = batch
            
            # Get predictions
            sentiments, probs = self.model.predict(processed_batch)
            
            # Create result dictionaries
            for j, review in enumerate(batch):
                result = {
                    'review': review,
                    'sentiment': sentiments[j],
                }
                
                # Add probability scores
                if hasattr(self.model, 'deep_model'):
                    classes = self.model.deep_model.label_encoder.classes_
                elif hasattr(self.model, 'label_encoder'):
                    classes = self.model.label_encoder.classes_
                else:
                    # Fallback for test models
                    if hasattr(self.model, 'classes_'):
                        classes = self.model.classes_
                    else:
                        # Default classes for testing
                        classes = np.array(['negative', 'neutral', 'positive'])
                    
                if len(probs.shape) > 1 and probs.shape[1] > 1:
                    # Multi-class case
                    result['confidence'] = float(np.max(probs[j]))
                    for k, class_name in enumerate(classes):
                        result[f'{class_name}_score'] = float(probs[j][k])
                else:
                    # Binary case
                    if len(probs.shape) == 1:
                        pos_prob = float(probs[j])
                    else:
                        pos_prob = float(probs[j][0])
                        
                    result['positive_score'] = pos_prob
                    result['negative_score'] = 1 - pos_prob
                    result['confidence'] = max(pos_prob, 1 - pos_prob)
                
                results.append(result)
        
        return results
    
    def analyze_misclassifications(self, df, text_column, true_label_column, pred_label_column):
        """Analyze misclassified reviews
        
        Args:
            df (pd.DataFrame): DataFrame with reviews
            text_column (str): Column containing text
            true_label_column (str): Column with true labels
            pred_label_column (str): Column with predicted labels
            
        Returns:
            pd.DataFrame: Misclassified reviews with analysis
        """
        # Find misclassified reviews
        misclassified = df[df[true_label_column] != df[pred_label_column]].copy()
        
        if len(misclassified) == 0:
            print("No misclassifications found!")
            return pd.DataFrame()
        
        print(f"Found {len(misclassified)} misclassified reviews")
        
        # Analyze length of misclassified reviews
        misclassified['text_length'] = misclassified[text_column].apply(len)
        misclassified['word_count'] = misclassified[text_column].apply(lambda x: len(str(x).split()))
        
        # Analyze error transitions
        error_transitions = misclassified.groupby([true_label_column, pred_label_column]).size()
        print("\nError transitions:")
        print(error_transitions)
        
        # Plot error transitions if we have multiple error types
        if len(error_transitions) > 1:
            error_df = pd.DataFrame(error_transitions).reset_index()
            error_df.columns = ['True Label', 'Predicted Label', 'Count']
            
            plt.figure(figsize=(10, 8))
            pivot_table = error_df.pivot_table(
                index='True Label', 
                columns='Predicted Label', 
                values='Count',
                fill_value=0
            )
            
            sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Reds')
            plt.title('Misclassification Patterns')
            plt.tight_layout()
            plt.savefig('visualizations/misclassification_patterns.png')
            plt.show()
        
        # Find words that are common in misclassifications
        all_words = " ".join(misclassified[text_column]).lower().split()
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(20)
        
        print("\nMost common words in misclassified reviews:")
        for word, count in most_common:
            print(f"  {word}: {count}")
        
        # Return examples of misclassifications
        print("\nExamples of misclassifications:")
        examples = misclassified.head(min(5, len(misclassified)))
        for i, (_, row) in enumerate(examples.iterrows()):
            print(f"\nReview {i+1}: {row[text_column][:100]}...")
            print(f"True sentiment: {row[true_label_column]}")
            print(f"Predicted sentiment: {row[pred_label_column]}")
        
        return misclassified[[text_column, true_label_column, pred_label_column, 'text_length', 'word_count']]
    
    def create_interactive_analyzer(self):
        """Create an interactive analyzer for Jupyter/Colab
        
        This creates interactive widgets for analyzing reviews in Jupyter notebooks
        """
        # Try importing ipywidgets
        try:
            import ipywidgets as widgets
            from IPython.display import display, HTML
        except ImportError:
            print("Error: ipywidgets not installed. Run: !pip install ipywidgets")
            return
            
        # Create text area for input
        review_input = widgets.Textarea(
            value='',
            placeholder='Enter an Amazon review to analyze...',
            description='Review:',
            disabled=False,
            layout=widgets.Layout(width='100%', height='100px')
        )
        
        # Create output widget
        output = widgets.Output()
        
        # Create button
        button = widgets.Button(
            description='Analyze Sentiment',
            button_style='success',
            tooltip='Click to analyze sentiment',
            icon='check'
        )
        
        # Function to handle button click
        def on_button_click(b):
            with output:
                output.clear_output()
                
                if not review_input.value:
                    print("Please enter a review to analyze!")
                    return
                
                # Analyze review
                result = self.analyze_review(review_input.value)
                
                # Determine color based on sentiment
                if result['sentiment'] == 'positive':
                    color = 'green'
                elif result['sentiment'] == 'negative':
                    color = 'red'
                else:
                    color = 'orange'
                
                # Display formatted result
                display(HTML(f"""
                <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px; background-color: #f9f9f9;">
                    <h3>Sentiment Analysis Result:</h3>
                    <p><strong>Review:</strong> {result['review'][:100]}{"..." if len(result['review']) > 100 else ""}</p>
                    <p><strong>Sentiment:</strong> <span style="color: {color}; font-weight: bold;">{result['sentiment']}</span></p>
                    <p><strong>Confidence:</strong> {result.get('confidence', 0)*100:.2f}%</p>
                    
                    <h4>Score Details:</h4>
                    <ul>
                """))
                
                # Add score details
                for key, value in result.items():
                    if key.endswith('_score'):
                        display(HTML(f"<li><strong>{key.replace('_score', '')}:</strong> {value*100:.2f}%</li>"))
                
                display(HTML("</ul></div>"))
        
        # Register button click event
        button.on_click(on_button_click)
        
        # Display widgets
        display(widgets.HTML("<h2>Amazon Review Sentiment Analyzer</h2>"))
        display(review_input)
        display(button)
        display(output)
    
    def create_analysis_dashboard(self, df, text_column, true_label_column=None, pred_label_column=None):
        """Create a dashboard with insights from sentiment analysis
        
        Args:
            df (pd.DataFrame): DataFrame with reviews and predictions
            text_column (str): Column containing review text
            true_label_column (str, optional): Column with true labels
            pred_label_column (str, optional): Column with predicted labels
        """
        print("=== Sentiment Analysis Dashboard ===")
        
        # If we have true and predicted labels, calculate accuracy
        if true_label_column and pred_label_column and true_label_column in df.columns and pred_label_column in df.columns:
            accuracy = np.mean(df[true_label_column] == df[pred_label_column])
            print(f"\nModel Accuracy: {accuracy:.4f}")
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(df[true_label_column], df[pred_label_column])
            unique_labels = sorted(df[true_label_column].unique())
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=unique_labels,
                       yticklabels=unique_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('visualizations/dashboard_confusion_matrix.png')
            plt.show()
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(df[true_label_column], df[pred_label_column]))
        
        # Get sentiment distribution
        if pred_label_column and pred_label_column in df.columns:
            sentiment_counts = df[pred_label_column].value_counts()
        elif true_label_column and true_label_column in df.columns:
            sentiment_counts = df[true_label_column].value_counts()
        else:
            print("No sentiment labels found in dataframe.")
            return
            
        # Plot sentiment distribution
        plt.figure(figsize=(10, 6))
        ax = sentiment_counts.plot(kind='bar', color=['green', 'orange', 'red'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        
        # Add percentage labels on top of bars
        total = sentiment_counts.sum()
        for i, count in enumerate(sentiment_counts):
            percentage = count / total * 100
            ax.text(i, count + 0.1, f'{percentage:.1f}%', ha='center')
            
        plt.tight_layout()
        plt.savefig('visualizations/dashboard_sentiment_distribution.png')
        plt.show()
        
        # Word clouds by sentiment
        sentiments = sorted(sentiment_counts.index)
        if len(sentiments) > 0:
            fig, axes = plt.subplots(1, len(sentiments), figsize=(6*len(sentiments), 6))
            
            # Adjust for single sentiment case
            if len(sentiments) == 1:
                axes = [axes]
                
            for i, sentiment in enumerate(sentiments):
                # Get column to filter by
                filter_col = pred_label_column if pred_label_column in df.columns else true_label_column
                
                # Get texts for this sentiment
                texts = df[df[filter_col] == sentiment][text_column].values
                
                if len(texts) == 0:
                    axes[i].text(0.5, 0.5, f"No {sentiment} reviews found", 
                              horizontalalignment='center', verticalalignment='center')
                    axes[i].set_title(f'{sentiment.capitalize()} Reviews (None Found)')
                    axes[i].axis('off')
                    continue
                    
                # Combine all texts
                all_text = ' '.join(texts)
                
                # Create word cloud
                wordcloud = WordCloud(width=800, height=400, background_color='white',
                                    max_words=100, contour_width=3).generate(all_text)
                
                # Plot
                axes[i].imshow(wordcloud, interpolation='bilinear')
                axes[i].set_title(f'{sentiment.capitalize()} Reviews')
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.savefig('visualizations/dashboard_wordclouds.png')
            plt.show()
        
        # Review length analysis
        plt.figure(figsize=(12, 6))
        df['word_count'] = df[text_column].apply(lambda x: len(str(x).split()))
        
        if true_label_column in df.columns:
            sns.boxplot(x=true_label_column, y='word_count', data=df)
            plt.title('Review Word Count by True Sentiment')
        elif pred_label_column in df.columns:
            sns.boxplot(x=pred_label_column, y='word_count', data=df)
            plt.title('Review Word Count by Predicted Sentiment')
        
        plt.xlabel('Sentiment')
        plt.ylabel('Word Count')
        plt.tight_layout()
        plt.savefig('visualizations/dashboard_word_count.png')
        plt.show()
        
        # Show some example reviews for each sentiment
        print("\nExample Reviews by Sentiment:")
        label_col = pred_label_column if pred_label_column in df.columns else true_label_column
        
        for sentiment in sentiments:
            sentiment_df = df[df[label_col] == sentiment]
            examples = sentiment_df.sample(min(3, len(sentiment_df)))
            
            print(f"\n{sentiment.upper()} REVIEWS:")
            for i, (_, row) in enumerate(examples.iterrows()):
                print(f"  {i+1}. {row[text_column][:100]}...")

# Test function
def test_analyzer():
    """Test the sentiment analyzer with dummy data"""
    # Import necessary modules
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    # Create a simple model class for testing
    class DummyModel:
        def __init__(self):
            # Add label_encoder with classes
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['negative', 'neutral', 'positive'])
            # Or alternatively:
            self.classes_ = np.array(['negative', 'neutral', 'positive'])
            
        def predict(self, texts):
            n = len(texts)
            sentiments = ["positive"] * n
            
            # Generate random probabilities
            probs = np.random.random((n, 3))
            # Make positive probability higher
            probs[:, 2] = 0.7 + 0.3 * np.random.random(n)
            # Normalize
            probs = probs / probs.sum(axis=1, keepdims=True)
            
            return sentiments, probs
    
    # Create sample data
    texts = [
        "This product is amazing! I love it so much.",
        "Terrible quality, broke after one day. Would not recommend!",
        "It's okay, not great but not bad either. Average product."
    ]
    
    # Create dummy model and analyzer
    model = DummyModel()
    analyzer = SentimentAnalyzer(model)
    
    # Test analyze_review
    result = analyzer.analyze_review(texts[0])
    print("Single review analysis:")
    print(result)
    
    # Test analyze_batch
    batch_results = analyzer.analyze_batch(texts)
    print("\nBatch analysis:")
    for result in batch_results:
        print(f"Review: {result['review'][:30]}..., Sentiment: {result['sentiment']}")

if __name__ == "__main__":
    test_analyzer()