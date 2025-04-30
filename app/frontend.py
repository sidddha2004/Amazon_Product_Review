import os
import sys
import requests
import gradio as gr

# Default API URL (can be customized)
API_URL = "http://localhost:5000/api"

def analyze_review(review_text):
    """
    Send a review to the API for analysis
    """
    try:
        response = requests.post(
            f"{API_URL}/analyze_review",
            json={"review": review_text}
        )
        
        if response.status_code == 200:
            result = response.json()
            return format_results(result)
        else:
            return f"Error: {response.text}"
    
    except Exception as e:
        return f"Error: {str(e)}"

def format_results(result):
    """
    Format API results for display
    """
    formatted = []
    
    # Original review
    formatted.append(f"### Original Review\n{result['review']}")
    
    # Sentiment
    if 'sentiment' in result['analysis']:
        sentiment = result['analysis']['sentiment']
        formatted.append(f"### Sentiment\n**{sentiment['label'].capitalize()}**")
        
        if 'scores' in sentiment:
            scores = sentiment['scores']
            score_text = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
            formatted.append(f"Scores: {score_text}")
    
    # Aspects
    if 'aspects' in result['analysis']:
        aspects = result['analysis']['aspects']
        formatted.append("### Aspect Sentiments")
        
        for aspect, data in aspects.items():
            if isinstance(data, dict) and 'sentiment' in data:
                formatted.append(f"- **{aspect}**: {data['sentiment']}")
            else:
                formatted.append(f"- **{aspect}**: {data}")
    
    # Helpfulness
    if 'helpfulness' in result['analysis']:
        helpfulness = result['analysis']['helpfulness']
        is_helpful = helpfulness.get('is_helpful', False)
        
        helpful_text = "Likely helpful" if is_helpful else "May not be helpful"
        
        if 'probability' in helpfulness:
            prob = helpfulness['probability']
            helpful_text += f" (confidence: {prob:.2f})"
        
        formatted.append(f"### Helpfulness\n{helpful_text}")
    
    # Summary
    if 'summary' in result['analysis']:
        summary = result['analysis']['summary']
        formatted.append(f"### Summary\n{summary}")
    
    return "\n\n".join(formatted)

def create_interface():
    """
    Create the Gradio interface
    """
    # Check if API is available
    try:
        response = requests.get(f"{API_URL}/models")
        api_available = response.status_code == 200
        model_info = response.json() if api_available else {}
    except:
        api_available = False
        model_info = {}
    
    # Create interface
    review_input = gr.Textbox(
        lines=10,
        placeholder="Enter a product review here...",
        label="Product Review"
    )
    
    analysis_output = gr.Markdown(
        label="Analysis Results"
    )
    
    interface = gr.Interface(
        fn=analyze_review,
        inputs=review_input,
        outputs=analysis_output,
        title="Amazon Product Review Analysis",
        description="Analyze Amazon product reviews for sentiment, aspects, helpfulness, and generate summaries.",
        examples=[
            ["This product is amazing! The battery life is excellent and the design is beautiful."],
            ["I'm very disappointed with this purchase. It broke after just one week of use."],
            ["The quality is good for the price, but the customer service was not helpful when I had an issue."]
        ]
    )
    
    # Add API status
    if not api_available:
        interface.description += "\n\n**⚠️ Warning: API not available. Make sure the API is running.**"
    else:
        models_available = ", ".join(model_info.keys())
        interface.description += f"\n\nModels available: {models_available}"
    
    return interface

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Start the Amazon Review Analysis Web Interface')
    parser.add_argument('--api_url', type=str, default='http://localhost:5000/api', help='URL of the API')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the web interface on')
    
    args = parser.parse_args()
    
    # Set API URL
    API_URL = args.api_url
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(server_port=args.port)