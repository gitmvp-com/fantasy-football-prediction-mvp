import torch
import numpy as np
import pandas as pd

def predict(model, X):
    """
    Generate predictions using the trained model.
    
    Args:
        model: Trained LSTM model
        X: Input features (numpy array)
    
    Returns:
        Predictions as numpy array
    """
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return predictions

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    predictions = predict(model, X_test)
    predictions = predictions.flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - y_test.values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test.values))
    
    # R-squared
    ss_res = np.sum((y_test.values - predictions) ** 2)
    ss_tot = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }
    
    return metrics
