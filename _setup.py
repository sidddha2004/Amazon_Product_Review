# 1_setup.py - Setup and utilities for Amazon Review Sentiment Analysis

# Install required packages
import sys
!pip install -q pandas nltk scikit-learn matplotlib seaborn tensorflow tqdm wordcloud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input, Bidirectional, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import gc  # Garbage collection for memory management
import time
from tqdm.notebook import tqdm

print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Create necessary directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Display information about GPU availability
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU Available: {len(gpus)} GPU(s)")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("No GPU found. Running on CPU.")

# Memory management utility
def clear_memory():
    """Clear memory to prevent OOM errors"""
    gc.collect()
    tf.keras.backend.clear_session()
    print("Memory cleared!")

print("Setup complete!")