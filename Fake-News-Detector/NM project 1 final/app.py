import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import preprocess_text
from predict import predict_single_text

# Set page configuration
st.set_page_config(
    page_title="Fake News Detection on Reddit Posts",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load('models/best_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

def main():
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Analyzer", "Dashboard", "Predictions", "About"])
    
    # Load model and vectorizer
    try:
        model, vectorizer = load_model()
        model_loaded = True
    except:
        model_loaded = False
        if page not in ["Home", "About"]:
            st.error("Model files not found. Please run the data preprocessing and model training scripts first.")
    
    # Home page
    if page == "Home":
        st.title("Fake News Detection on Reddit Posts")
        st.markdown("### Using Machine Learning to Identify Fake News")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            This application uses machine learning to detect fake news in Reddit posts. 
            
            **Key Features:**
            - Text preprocessing using NLTK
            - Feature extraction using TF-IDF
            - Machine learning classification models
            - Interactive web interface for real-time analysis
            
            Use the sidebar to navigate to different sections of the application.
            """)
        
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/2965/2965879.png", width=200)
        
        st.markdown("---")
        
        st.markdown("""
        ### How to use this application:
        
        1. **Analyzer**: Input a Reddit post to check if it's real or fake news
        2. **Dashboard**: View model performance metrics and visualizations
        3. **Predictions**: See predictions made on the test dataset
        4. **About**: Learn more about the project and methodology
        """)
    
    # Analyzer page
    elif page == "Analyzer" and model_loaded:
        st.title("Reddit Post Analyzer")
        st.markdown("Enter a Reddit post to check if it's real or fake news.")
        
        # Text input
        text_input = st.text_area("Post Text", height=150, placeholder="Enter the Reddit post text here...")
        
        # Analyze button
        if st.button("Analyze"):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    # Make prediction
                    result = predict_single_text(text_input, model, vectorizer)
                    
                    # Display result
                    st.markdown("---")
                    st.subheader("Analysis Result")
                    
                    # Create columns for result
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        if result['prediction'] == 1:
                            st.success("**Prediction: Real News**")
                            st.markdown("This post is classified as real news with high confidence.")
                        else:
                            st.error("**Prediction: Fake News**")
                            st.markdown("This post is classified as fake news with high confidence.")
                    
                    with col2:
                        # Create a DataFrame for the probabilities
                        probs_df = pd.DataFrame({
                            'Category': ['Real News', 'Fake News'],
                            'Probability': [result['real_probability'], result['fake_probability']]
                        })
                        
                        # Plot the probabilities
                        fig, ax = plt.subplots(figsize=(8, 4))
                        bars = ax.barh(probs_df['Category'], probs_df['Probability'], color=['green', 'red'])
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Probability')
                        ax.set_title('Prediction Probabilities')
                        
                        # Add text labels
                        for bar in bars:
                            width = bar.get_width()
                            label_x_pos = width if width > 0.05 else 0.05
                            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                                   va='center', color='white' if width > 0.3 else 'black')
                        
                        st.pyplot(fig)
            else:
                st.warning("Please enter some text to analyze.")
    
    # Dashboard page
    elif page == "Dashboard" and model_loaded:
        st.title("Model Performance Dashboard")
        st.markdown("View metrics and visualizations from the fake news detection model.")
        
        # Check if visualizations exist
        visualizations = []
        if os.path.exists('static'):
            visualizations = [f for f in os.listdir('static') if f.endswith('.png')]
        
        if visualizations:
            st.markdown("### Model Visualizations")
            
            # Create a selectbox for visualizations
            selected_viz = st.selectbox(
                "Select Visualization", 
                options=visualizations,
                format_func=lambda x: x.replace('_', ' ').replace('.png', '').title()
            )
            
            # Display the selected visualization
            if selected_viz:
                st.image(f"static/{selected_viz}", caption=selected_viz.replace('_', ' ').replace('.png', '').title())
        
        # Model explanation
        st.markdown("---")
        st.markdown("### Model Explanation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### What is TF-IDF?")
            st.markdown("""
            Term Frequency-Inverse Document Frequency is a numerical statistic that reflects how important a word is to a document in a collection. 
            It helps the model identify distinctive words that characterize fake vs. real news.
            """)
        
        with col2:
            st.markdown("#### Feature Importance")
            st.markdown("""
            The model identifies certain words and phrases that are strong indicators of fake or real news. 
            These features help the model make accurate predictions.
            """)
        
        with col3:
            st.markdown("#### Model Evaluation")
            st.markdown("""
            The model is evaluated using metrics like accuracy, precision, recall, and F1 score. 
            These metrics help us understand how well the model performs on new data.
            """)
        
        # How it works
        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown("""
        The fake news detection system works through the following steps:
        
        1. **Text Preprocessing:** Clean and normalize text by removing special characters, converting to lowercase, and removing stopwords.
        2. **Feature Extraction:** Convert text to numerical features using TF-IDF vectorization.
        3. **Model Training:** Train multiple machine learning models on the labeled data.
        4. **Model Selection:** Select the best performing model based on evaluation metrics.
        5. **Prediction:** Use the selected model to predict whether new posts are real or fake.
        
        The system also provides probability scores to indicate the confidence level of each prediction.
        """)
    
    # Predictions page
    elif page == "Predictions" and model_loaded:
        st.title("Test Data Predictions")
        st.markdown("View predictions made on the test dataset.")
        
        # Check if predictions file exists
        if os.path.exists('predictions.csv'):
            # Load predictions
            predictions_df = pd.read_csv('predictions.csv')
            
            # Load original test data to get the text
            test_data = pd.read_csv('x_test.csv')
            
            # Merge predictions with text
            merged_df = predictions_df.merge(test_data, on='ID')
            
            # Display the predictions
            st.dataframe(
                merged_df[['ID', 'text', 'predicted_class', 'real_probability', 'fake_probability']],
                column_config={
                    'ID': 'ID',
                    'text': 'Post Text',
                    'predicted_class': st.column_config.TextColumn('Prediction', help="Model's prediction"),
                    'real_probability': st.column_config.ProgressColumn(
                        'Real Probability',
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    'fake_probability': st.column_config.ProgressColumn(
                        'Fake Probability',
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Summary statistics
            st.markdown("---")
            st.subheader("Prediction Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Count of predictions by class
                class_counts = merged_df['predicted_class'].value_counts().reset_index()
                class_counts.columns = ['Class', 'Count']
                
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(class_counts['Class'], class_counts['Count'], color=['red', 'green'])
                ax.set_ylabel('Count')
                ax.set_title('Predictions by Class')
                
                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', 
                           ha='center', va='bottom')
                
                st.pyplot(fig)
            
            with col2:
                # Distribution of probabilities
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(merged_df['real_probability'], bins=20, alpha=0.7, label='Real Probability')
                ax.hist(merged_df['fake_probability'], bins=20, alpha=0.7, label='Fake Probability')
                ax.set_xlabel('Probability')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Prediction Probabilities')
                ax.legend()
                
                st.pyplot(fig)
        else:
            st.warning("No predictions available. Run the prediction script first to generate predictions on the test data.")
            st.code("python predict.py")
    
    # About page
    elif page == "About":
        st.title("About This Project")
        st.markdown("""
        ### Fake News Detection on Reddit Posts
        
        This project implements a machine learning model to detect fake news in Reddit posts. It includes data preprocessing, feature extraction, model training, and a web interface for demonstration.
        
        #### Project Structure
        
        - `data_preprocessing.py`: Handles data cleaning and preparation
        - `feature_extraction.py`: Extracts features from text data
        - `model_training.py`: Trains and evaluates the machine learning model
        - `predict.py`: Makes predictions on new data
        - `app.py`: Streamlit web application for demonstration
        - `models/`: Directory to store trained models
        - `utils.py`: Utility functions used across the project
        
        #### Dataset
        
        The project uses two datasets:
        - `xy_train.csv`: Training data with labels (real/fake)
        - `x_test.csv`: Test data for prediction
        
        #### Model
        
        The system uses a machine learning pipeline with TF-IDF vectorization and a classifier (Logistic Regression, Random Forest, or SVM) to detect fake news.
        
        #### Setup and Installation
        
        1. Install the required packages:
           ```bash
           pip install -r requirements.txt
           ```
        
        2. Download NLTK data:
           ```bash
           python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
           ```
        
        3. Run the data preprocessing and model training:
           ```bash
           python data_preprocessing.py
           python feature_extraction.py
           python model_training.py
           ```
        
        4. Start the Streamlit application:
           ```bash
           streamlit run app.py
           ```""")

if __name__ == "__main__":
    main()
