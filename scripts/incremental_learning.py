import pandas as pd
import tensorflow as tf
from keras.models import load_model
import joblib
from data_preprocessing import load_data

def load_trained_model():
    try:
        model = load_model('../models/trained_genai_model.h5')
        scaler = joblib.load('../models/scaler.pkl')
    except FileNotFoundError:
        print("Error: Model or scaler file not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the model or scaler: {e}")
        return None, None
    return model, scaler

def incremental_learning(new_data_file):
    # Load existing model and scaler
    model, scaler = load_trained_model()
    
    if model is None or scaler is None:
        print("Error: Loading model or scaler failed. Incremental learning aborted.")
        return

    # Load and scale new data
    new_data, _, _ = load_data(new_data_file)
    
    if new_data is None:
        print("Error: Loading new data failed. Incremental learning aborted.")
        return

    new_data_scaled = scaler.transform(new_data)

    try:
        # Retrain model with new data
        model.fit(new_data_scaled, new_data_scaled, epochs=10, batch_size=8)
        model.save('../models/trained_genai_model.h5')
        print("Model updated with new data.")
    except Exception as e:
        print(f"An error occurred during incremental learning: {e}")

if __name__ == "__main__":
    incremental_learning('../data/ML-Dataset.csv')
