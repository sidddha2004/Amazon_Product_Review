# Create a setup_nlp.py file
import nltk
import spacy

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Download SpaCy model
spacy.cli.download('en_core_web_sm')

print("NLP resources downloaded successfully!")