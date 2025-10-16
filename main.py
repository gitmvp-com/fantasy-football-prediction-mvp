import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

from data.sample_data import generate_sample_data, add_rolling_features
from scripts.model import FantasyFootballLSTM
from scripts.training import train_model
from scripts.prediction import predict, evaluate_model

def main():
    print("="*60)
    print("Fantasy Football Prediction MVP - LSTM Model")
    print("="*60)
    
    # Generate sample data
    print("\n1. Generating sample data...")
    df = generate_sample_data(num_players=100, num_years=5)
    df = add_rolling_features(df)
    
    # Remove rows without target (last year for each player)
    df_train = df[df['NextYearFantPt/G'] > 0].copy()
    df_2024 = df[df['Year'] == 2024].copy()
    
    print(f"   - Total records: {len(df_train)}")
    print(f"   - Players: {df_train['Player'].nunique()}")
    
    # Define features and target
    feature_names = [
        'Year', 'Age', 'G', 'Tgt', 'Rec', 'RecYds', 'RecTD', 
        'TD/G', 'RecYds/G', 'FantPtHalf/G', 'Tgt/G', 
        'FantPtHalf/GLast2Y', 'Tgt/GLast2Y', 'RecYds/GLast2Y', '#ofY'
    ]
    target = 'NextYearFantPt/G'
    
    # Split data
    print("\n2. Preparing train/validation/test splits...")
    X = df_train[feature_names]
    y = df_train[target]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Prepare 2024 data for predictions
    X_2024 = df_2024[feature_names]
    player_names_2024 = df_2024['Player'].values
    
    # Standardize features
    print("\n3. Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_2024_scaled = scaler.transform(X_2024)
    
    # Reshape for LSTM (samples, timesteps, features)
    X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_val_scaled = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
    X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    X_2024_scaled = X_2024_scaled.reshape(X_2024_scaled.shape[0], 1, X_2024_scaled.shape[1])
    
    # Create and train model
    print("\n4. Training LSTM model...")
    input_dim = X_train_scaled.shape[2]
    hidden_dim = 128
    num_layers = 2
    
    model = FantasyFootballLSTM(input_dim, hidden_dim, num_layers)
    model = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val, epochs=200, learning_rate=0.001)
    
    # Evaluate model
    print("\n5. Evaluating model on test set...")
    metrics = evaluate_model(model, X_test_scaled, y_test)
    print(f"   - RMSE: {metrics['RMSE']:.3f}")
    print(f"   - MAE: {metrics['MAE']:.3f}")
    print(f"   - R²: {metrics['R2']:.3f}")
    
    # Make predictions for 2025
    print("\n6. Generating 2025 predictions...")
    predictions_2025 = predict(model, X_2024_scaled).flatten()
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'Player': player_names_2024,
        'Predicted_FantPt/G_2025': np.round(predictions_2025, 1)
    })
    
    # Sort by predicted points
    predictions_df = predictions_df.sort_values('Predicted_FantPt/G_2025', ascending=False).reset_index(drop=True)
    
    # Display top 10 predictions
    print("\n" + "="*60)
    print("TOP 10 PREDICTED PLAYERS FOR 2025")
    print("="*60)
    print(predictions_df.head(10).to_string(index=True))
    
    print("\n" + "="*60)
    print("MVP run complete!")
    print("="*60)

if __name__ == "__main__":
    main()
