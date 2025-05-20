import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import plot_confusion_matrix, plot_feature_importance, get_model_metrics, save_metrics_plot

def load_features(sample_size=None):
    """
    Load the extracted features and labels
    Optional: Load only a sample of the data for faster training
    """
    X_train_tfidf = joblib.load('models/X_train_tfidf.pkl')
    y_train = joblib.load('models/y_train.pkl')
    
    # Optionally sample a subset of data for faster training
    if sample_size and sample_size < X_train_tfidf.shape[0]:
        indices = np.random.choice(X_train_tfidf.shape[0], sample_size, replace=False)
        X_train_tfidf = X_train_tfidf[indices]
        y_train = y_train[indices]
        print(f"Using a sample of {sample_size} examples for faster training")
    
    print(f"Loaded training features: {X_train_tfidf.shape}")
    print(f"Loaded training labels: {y_train.shape}")
    
    return X_train_tfidf, y_train

def split_validation_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and validation sets
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Train a Logistic Regression model with optimized parameters
    """
    print("\nTraining Logistic Regression model...")
    
    # Simplified parameter grid for faster training
    param_grid = {
        'C': [1.0],  # Reduced from [0.1, 1.0, 10.0]
        'penalty': ['l2'],  # Reduced from ['l1', 'l2']
        'solver': ['liblinear']
    }
    
    # Initialize model
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Perform grid search with fewer CV folds
    grid_search = GridSearchCV(
        lr_model, param_grid, cv=3, scoring='accuracy', n_jobs=4  # Reduced CV folds and specific n_jobs
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_lr_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_pred = best_lr_model.predict(X_val)
    
    # Get metrics
    metrics = get_model_metrics(y_val, y_pred)
    print("\nLogistic Regression Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['report'])
    
    # Save model
    joblib.dump(best_lr_model, 'models/logistic_regression_model.pkl')
    print("Logistic Regression model saved to 'models/logistic_regression_model.pkl'")
    
    return best_lr_model, metrics

def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train a Random Forest model with optimized parameters
    """
    print("\nTraining Random Forest model...")
    
    # Simplified parameter grid for faster training
    param_grid = {
        'n_estimators': [100],  # Reduced from [100, 200]
        'max_depth': [10],      # Reduced from [None, 10, 20]
        'min_samples_split': [2]  # Reduced from [2, 5]
    }
     
    # Initialize model
    rf_model = RandomForestClassifier(random_state=42)
    
    # Perform grid search with fewer CV folds
    grid_search = GridSearchCV(
        rf_model, param_grid, cv=3, scoring='accuracy', n_jobs=4  # Reduced CV folds and specific n_jobs
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_pred = best_rf_model.predict(X_val)
    
    # Get metrics
    metrics = get_model_metrics(y_val, y_pred)
    print("\nRandom Forest Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['report'])
    
    # Save model
    joblib.dump(best_rf_model, 'models/random_forest_model.pkl')
    print("Random Forest model saved to 'models/random_forest_model.pkl'")
    
    return best_rf_model, metrics

def train_svm(X_train, y_train, X_val, y_val):
    """
    Train an SVM model with optimized parameters
    """
    print("\nTraining SVM model...")
    
    # Simplified parameter grid for faster training
    param_grid = {
        'C': [1.0],        # Reduced from [0.1, 1.0, 10.0]
        'kernel': ['linear'],  # Reduced from ['linear', 'rbf']
        'gamma': ['scale']     # Reduced from ['scale', 'auto']
    }
    
    # Initialize model
    svm_model = SVC(probability=True, random_state=42)
    
    # Perform grid search with fewer CV folds
    grid_search = GridSearchCV(
        svm_model, param_grid, cv=3, scoring='accuracy', n_jobs=4  # Reduced CV folds and specific n_jobs
    )
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_svm_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate on validation set
    y_pred = best_svm_model.predict(X_val)
    
    # Get metrics
    metrics = get_model_metrics(y_val, y_pred)
    print("\nSVM Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("\nClassification Report:")
    print(metrics['report'])
    
    # Save model
    joblib.dump(best_svm_model, 'models/svm_model.pkl')
    print("SVM model saved to 'models/svm_model.pkl'")
    
    return best_svm_model, metrics

def visualize_results(lr_model, rf_model, svm_model, lr_metrics, rf_metrics, svm_metrics, X_val, y_val, feature_names):
    """
    Visualize model results and comparisons
    """
    print("\nGenerating visualizations...")
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Plot confusion matrices
    # Get unique classes
    unique_classes = np.unique(y_val)
    if len(unique_classes) == 2:
        classes = ['Fake', 'Real']
    else:
        # For multiclass, use numeric labels
        classes = [str(c) for c in unique_classes]
    
    # Logistic Regression confusion matrix
    lr_y_pred = lr_model.predict(X_val)
    lr_cm = plot_confusion_matrix(y_val, lr_y_pred, classes)
    lr_cm.savefig('static/lr_confusion_matrix.png')
    plt.close()
    
    # Random Forest confusion matrix
    rf_y_pred = rf_model.predict(X_val)
    rf_cm = plot_confusion_matrix(y_val, rf_y_pred, classes)
    rf_cm.savefig('static/rf_confusion_matrix.png')
    plt.close()
    
    # SVM confusion matrix
    svm_y_pred = svm_model.predict(X_val)
    svm_cm = plot_confusion_matrix(y_val, svm_y_pred, classes)
    svm_cm.savefig('static/svm_confusion_matrix.png')
    plt.close()
    
    # Plot feature importance for Random Forest
    if hasattr(rf_model, 'feature_importances_'):
        rf_fi = plot_feature_importance(rf_model, feature_names, top_n=20)
        rf_fi.savefig('static/rf_feature_importance.png')
        plt.close()
    
    # Compare model performance
    models = ['Logistic Regression', 'Random Forest', 'SVM']
    metrics_plot = save_metrics_plot(
        models,
        [lr_metrics, rf_metrics, svm_metrics],
        'model_comparison.png'
    )
    plt.close()
    
    print("Visualizations saved to 'static/' directory")

def select_best_model(lr_metrics, rf_metrics, svm_metrics):
    """
    Select the best performing model based on F1 score
    """
    models = {
        'Logistic Regression': (lr_metrics['f1_score'], 'models/logistic_regression_model.pkl'),
        'Random Forest': (rf_metrics['f1_score'], 'models/random_forest_model.pkl'),
        'SVM': (svm_metrics['f1_score'], 'models/svm_model.pkl')
    }
    
    best_model_name = max(models.items(), key=lambda x: x[1][0])[0]
    best_model_path = models[best_model_name][1]
    
    print(f"\nBest model based on F1 score: {best_model_name}")
    
    # Create a symbolic link or copy to best_model.pkl
    best_model = joblib.load(best_model_path)
    joblib.dump(best_model, 'models/best_model.pkl')
    
    print("Best model saved as 'models/best_model.pkl'")
    
    return best_model_name

def main(fast_mode=True):
    print("Starting model training...")
    
    # Load features with optional sampling for faster training
    sample_size = 5000 if fast_mode else None
    X_train_tfidf, y_train = load_features(sample_size)
    
    # Load feature names
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Split data for validation
    X_train, X_val, y_train_split, y_val = split_validation_data(X_train_tfidf, y_train)
    
    # Train models
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train_split, X_val, y_val)
    rf_model, rf_metrics = train_random_forest(X_train, y_train_split, X_val, y_val)
    svm_model, svm_metrics = train_svm(X_train, y_train_split, X_val, y_val)
    
    # Visualize results
    visualize_results(
        lr_model, rf_model, svm_model,
        lr_metrics, rf_metrics, svm_metrics,
        X_val, y_val, feature_names
    )
    
    # Select best model
    best_model = select_best_model(lr_metrics, rf_metrics, svm_metrics)
    
    print("Model training completed successfully.")
    if fast_mode:
        print("\nNOTE: Models were trained in fast mode with reduced parameters and data.")
        print("For production use, set fast_mode=False in the main function call.")

if __name__ == "__main__":
    main(fast_mode=True)  # Set to True for faster training, False for full training
