from flask import Flask, render_template, request, jsonify
import joblib
import os
import pandas as pd
import numpy as np
from utils import preprocess_text
from predict import predict_single_text

app = Flask(__name__)

# Load model and vectorizer
def load_model():
    model = joblib.load('models/best_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

try:
    model, vectorizer = load_model()
    model_loaded = True
except:
    model_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Check if visualizations exist
    visualizations = []
    if os.path.exists('static'):
        visualizations = [f for f in os.listdir('static') if f.endswith('.png')]
    
    return render_template('dashboard.html', visualizations=visualizations)

@app.route('/predictions')
def predictions():
    predictions_exist = os.path.exists('predictions.csv')
    predictions_data = []
    
    if predictions_exist:
        # Load predictions
        predictions_df = pd.read_csv('predictions.csv')
        
        # Load original test data to get the text
        test_data = pd.read_csv('x_test.csv')
        
        # Merge predictions with text
        merged_df = predictions_df.merge(test_data, on='ID')
        
        # Convert to list of dictionaries for template
        predictions_data = merged_df.to_dict('records')
    
    return render_template('predictions.html', predictions_exist=predictions_exist, predictions=predictions_data)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Please run the model training scripts first.'}), 500
    
    data = request.get_json()
    text = data.get('text', '')
    
    if not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    # Make prediction
    result = predict_single_text(text, model, vectorizer)
    
    return jsonify({
        'prediction': int(result['prediction']),
        'real_probability': float(result['real_probability']),
        'fake_probability': float(result['fake_probability'])
    })

if __name__ == '__main__':
    app.run(debug=True)
