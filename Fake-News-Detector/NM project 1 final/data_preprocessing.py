import os
import pandas as pd
from utils import preprocess_text
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """
    Load training and test datasets
    """
    train_data = pd.read_csv('xy_train.csv')
    test_data = pd.read_csv('x_test.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data

def explore_data(train_data):
    """
    Explore and visualize the dataset
    """
    # Create directory for visualizations
    os.makedirs('static', exist_ok=True)
    
    # Check class distribution
    class_distribution = train_data['label'].value_counts()
    print("Class distribution:")
    print(class_distribution)
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=train_data, palette=['red', 'green'])
    plt.title('Class Distribution (0: Fake, 1: Real)')
    plt.xlabel('News Type')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Fake', 'Real'])
    plt.savefig('static/class_distribution.png')
    plt.close()
    
    # Text length distribution
    train_data['text_length'] = train_data['text'].apply(lambda x: len(str(x)))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_data, x='text_length', hue='label', bins=50, kde=True)
    plt.title('Text Length Distribution by Class')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.legend(['Fake', 'Real'])
    plt.savefig('static/text_length_distribution.png')
    plt.close()
    
    # Get some statistics
    print("\nText length statistics:")
    print(train_data.groupby('label')['text_length'].describe())
    
    return class_distribution

def preprocess_data(train_data, test_data):
    """
    Preprocess the text data
    """
    print("Preprocessing training data...")
    train_data['processed_text'] = train_data['text'].apply(preprocess_text)
    
    print("Preprocessing test data...")
    test_data['processed_text'] = test_data['text'].apply(preprocess_text)
    
    # Save preprocessed data
    train_data.to_csv('preprocessed_train.csv', index=False)
    test_data.to_csv('preprocessed_test.csv', index=False)
    
    print("Preprocessing complete. Saved preprocessed data to CSV files.")
    
    return train_data, test_data

def main():
    print("Starting data preprocessing...")
    
    # Load data
    train_data, test_data = load_data()
    
    # Explore data
    explore_data(train_data)
    
    # Preprocess data
    preprocessed_train, preprocessed_test = preprocess_data(train_data, test_data)
    
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
