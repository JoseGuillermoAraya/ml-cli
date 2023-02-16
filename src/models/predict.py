import pandas as pd
from src.data.preprocess_data import preprocess_data
import joblib

def predict(input_data, model_path):
    """Loads a trained model and uses it to make predictions on new data"""
    
    # Load trained model
    model = joblib.load(model_path)
    
    # Load data
    data = pd.read_csv(input_data)

    # Preprocess data
    data = preprocess_data(data)
    
    # Make predictions
    predictions = model.predict(data)
    
    return predictions
