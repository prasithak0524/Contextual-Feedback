import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import generate_wordcloud

def load_preprocessed_data():
    """
    Load preprocessed training and test data
    """
    train_data = pd.read_csv('preprocessed_train.csv')
    test_data = pd.read_csv('preprocessed_test.csv')
    
    print(f"Loaded preprocessed training data: {train_data.shape}")
    print(f"Loaded preprocessed test data: {test_data.shape}")
    
    return train_data, test_data

def extract_features(train_data, test_data):
    """
    Extract TF-IDF features from preprocessed text
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Fill NaN values with empty string
    train_data['processed_text'] = train_data['processed_text'].fillna('')
    test_data['processed_text'] = test_data['processed_text'].fillna('')
    
    # Initialize TF-IDF vectorizer
    print("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit features to reduce dimensionality
        min_df=5,           # Minimum document frequency
        max_df=0.8,         # Maximum document frequency
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    # Fit and transform training data
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['processed_text'])
    
    # Transform test data
    X_test_tfidf = tfidf_vectorizer.transform(test_data['processed_text'])
    
    print(f"Training features shape: {X_train_tfidf.shape}")
    print(f"Test features shape: {X_test_tfidf.shape}")
    
    # Save the vectorizer for later use
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    print("TF-IDF vectorizer saved to 'models/tfidf_vectorizer.pkl'")
    
    # Get feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    return X_train_tfidf, X_test_tfidf, feature_names

def visualize_features(train_data):
    """
    Create visualizations for feature analysis
    """
    print("Generating word clouds for real and fake news...")
    
    # Generate word cloud for fake news (label=0)
    fake_wc = generate_wordcloud(
        train_data['processed_text'].tolist(),
        train_data['label'].tolist(),
        0
    )
    fake_wc.savefig('static/fake_news_wordcloud.png')
    fake_wc.close()
    
    # Generate word cloud for real news (label=1)
    real_wc = generate_wordcloud(
        train_data['processed_text'].tolist(),
        train_data['label'].tolist(),
        1
    )
    real_wc.savefig('static/real_news_wordcloud.png')
    real_wc.close()
    
    print("Word clouds saved to 'static/' directory")

def save_features(X_train_tfidf, X_test_tfidf, train_data, test_data):
    """
    Save extracted features and labels
    """
    # Save training features and labels
    joblib.dump(X_train_tfidf, 'models/X_train_tfidf.pkl')
    joblib.dump(train_data['label'].values, 'models/y_train.pkl')
    
    # Save test features and IDs
    joblib.dump(X_test_tfidf, 'models/X_test_tfidf.pkl')
    joblib.dump(test_data['ID'].values, 'models/test_ids.pkl')
    
    print("Features and labels saved to 'models/' directory")

def main():
    print("Starting feature extraction...")
    
    # Load preprocessed data
    train_data, test_data = load_preprocessed_data()
    
    # Extract features
    X_train_tfidf, X_test_tfidf, feature_names = extract_features(train_data, test_data)
    
    # Visualize features
    visualize_features(train_data)
    
    # Save features for model training
    save_features(X_train_tfidf, X_test_tfidf, train_data, test_data)
    
    print("Feature extraction completed successfully.")

if __name__ == "__main__":
    main()
