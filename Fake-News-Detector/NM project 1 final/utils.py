import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from wordcloud import WordCloud

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text(text):
    """
    Clean and preprocess text data
    """
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|\#\w+', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

def tokenize_text(text):
    """
    Tokenize text into words
    """
    return word_tokenize(text)

def remove_stopwords(tokens):
    """
    Remove stopwords from tokenized text
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    """
    Lemmatize tokens to their root form
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text):
    """
    Complete text preprocessing pipeline
    """
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)

def plot_confusion_matrix(y_true, y_pred, classes):
    """ 
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importance for models that support it
    """
    if hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        # For tree-based models like Random Forest
        importance = model.feature_importances_
    else:
        return None
    
    # Get top features
    indices = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importance, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Important Features')
    plt.tight_layout()
    return plt

def generate_wordcloud(texts, labels, label_value):
    """
    Generate word cloud for a specific class (real or fake)
    """
    # Filter texts by label
    filtered_texts = [text for text, label in zip(texts, labels) if label == label_value]
    
    # Join all texts
    text = ' '.join(filtered_texts)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, contour_width=3).generate(text)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {"Real" if label_value == 1 else "Fake"} News')
    plt.tight_layout()
    return plt

def get_model_metrics(y_true, y_pred):
    """
    Calculate and return model performance metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Check if binary or multiclass classification
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(unique_labels) > 2:
        # Multiclass
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        report = classification_report(y_true, y_pred)
    else:
        # Binary
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'])
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'report': report
    }
    
    return metrics

def save_metrics_plot(models_or_metrics, metrics_list=None, filename='model_metrics.png'):
    """
    Create and save a bar plot of model metrics
    Can be used in two ways:
    1. save_metrics_plot(metrics_dict) - for a single model
    2. save_metrics_plot(models_list, metrics_list) - for comparing multiple models
    """
    plt.figure(figsize=(10, 6))
    
    # Case 1: Single model metrics
    if metrics_list is None:
        metrics = models_or_metrics
        metrics_to_plot = {k: v for k, v in metrics.items() if k != 'report'}
        plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        plt.title('Model Performance Metrics')
    
    # Case 2: Multiple models comparison
    else:
        models = models_or_metrics
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric_name in enumerate(metrics_names):
            values = [metrics[metric_name] for metrics in metrics_list]
            plt.bar(x + width * (i - 1.5), values, width, label=metric_name.capitalize())
        
        plt.xticks(x, models)
        plt.legend()
        plt.title('Model Comparison')
    
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(f'static/{filename}')
    
    return plt.gcf()  # Return the figure for closing later
