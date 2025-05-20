
designed and developed bu GUNALAN

# Fake News Detection on Reddit Posts

This project implements a machine learning model to detect fake news in Reddit posts. It includes data preprocessing, feature extraction, model training, and a Streamlit web interface for demonstration.

## Project Structure 

- `data_preprocessing.py`: Handles data cleaning and preparation
- `feature_extraction.py`: Extracts features from text data
- `model_training.py`: Trains and evaluates the machine learning model
- `predict.py`: Makes predictions on new data
- `app.py`: Streamlit web application for demonstration
- `static/`: Directory for visualization images
- `models/`: Directory to store trained models
- `utils.py`: Utility functions used across the project

## Setup and Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Download NLTK data: (natural language tool kit)
   ```
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

3. Run the data preprocessing and model training:
   ```
   python data_preprocessing.py
   python feature_extraction.py
   python model_training.py
   ```

4. Start the Streamlit web application:
   ```
   streamlit run app.py
   ```

## Dataset

The project uses two datasets:
- `xy_train.csv`: Training data with labels (real/fake)
- `x_test.csv`: Test data for prediction

## Model

The system uses a machine learning pipeline with TF-IDF vectorization and a classifier (Logistic Regression, Random Forest, or SVM) to detect fake news.

## Streamlit Web Interface
 
The Streamlit web interface allows users to:
- Enter a Reddit post text to check if it's fake or real
- View model performance metrics and visualizations
- Explore feature importance and word clouds
- See predictions made on the test dataset
