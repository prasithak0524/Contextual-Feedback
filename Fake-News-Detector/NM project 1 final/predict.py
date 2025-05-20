import joblib
import pandas as pd
import numpy as np
from utils import preprocess_text

def load_model_and_vectorizer():
    """
    Load the trained model and TF-IDF vectorizer
    """
    # Load the best model
    model = joblib.load('models/best_model.pkl')
    print("Loaded model from 'models/best_model.pkl'")
    
    # Load the TF-IDF vectorizer
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    print("Loaded TF-IDF vectorizer from 'models/tfidf_vectorizer.pkl'")
    
    return model, vectorizer

def predict_single_text(text, model, vectorizer):
    """
    Make a prediction for a single text input
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform the text using the vectorizer
    text_features = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_features)[0]
    probability = model.predict_proba(text_features)[0]
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Real' if prediction == 1 else 'Fake',
        'real_probability': float(probability[1]),
        'fake_probability': float(probability[0])
    }
    
    return result

def predict_test_data():
    """
    Make predictions on the test dataset
    """
    print("Making predictions on test data...")
    
    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()
    
    # Load test features
    X_test = joblib.load('models/X_test_tfidf.pkl')
    test_ids = joblib.load('models/test_ids.pkl')
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Create results dataframe
    results = pd.DataFrame({
        'ID': test_ids,
        'predicted_label': predictions,
        'real_probability': probabilities[:, 1],
        'fake_probability': probabilities[:, 0]
    })
    
    # Map numeric labels to text labels
    results['predicted_class'] = results['predicted_label'].map({1: 'Real', 0: 'Fake'})
    
    # Save results to CSV
    results.to_csv('predictions.csv', index=False)
    print("Predictions saved to 'predictions.csv'")
    
    return results

def main():
    print("Starting prediction process...")
    
    # Make predictions on test data
    results = predict_test_data()
    
    # Print summary
    print("\nPrediction summary:")
    print(f"Total predictions: {len(results)}")
    print(f"Predicted real news: {sum(results['predicted_label'] == 1)}")
    print(f"Predicted fake news: {sum(results['predicted_label'] == 0)}")
    
    print("Prediction process completed successfully.")

if __name__ == "__main__":
    main()
