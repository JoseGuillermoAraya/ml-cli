import pandas as pd
import joblib

def predict(input_data, model_name):
    """Loads a trained model and uses it to make predictions on new data"""
    
    # Load trained model
    model = joblib.load(f"../../data/models/{model_name}")
    
    # Load input data
    data = pd.read_csv(input_data)
    
    # Make predictions
    predictions = model.predict(data)
    
    return predictions
