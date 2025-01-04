import tensorflow as tf
from keras import layers, models
from data_preprocessing import load_data
import joblib
import os

def build_genai_model(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(input_shape[0])  # Output shape matches input
    ])

    model.compile(optimizer='adam', loss='mse')
    return model

def train_genai_model(file_path):
    # Load data
    scaled_features, scaler, df = load_data(file_path)
    
    # Check if data loading was successful
    if scaled_features is None or scaler is None:
        print("Error: Data loading failed. Model training aborted.")
        return None

    # Build the model
    model = build_genai_model((scaled_features.shape[1],))

    try:
        # Train the model
        model.fit(scaled_features, scaled_features, epochs=50, batch_size=8, validation_split=0.2)

        # Ensure the models directory exists
        os.makedirs('../models', exist_ok=True)

        # Save the trained model and scaler
        model.save('../models/trained_genai_model.h5')
        joblib.dump(scaler, '../models/scaler.pkl')
        print("Model trained and saved.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        return None

    return model

if __name__ == "__main__":
    train_genai_model('../data/ML-Dataset.csv')
