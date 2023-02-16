import pandas as pd
import joblib

def predict(input_data, model_path):
    """Loads a trained model and uses it to make predictions on new data"""
    
    # Load trained model
    model = joblib.load(model_path)
    
    # Load input data
    data = pd.read_csv(input_data)
    
    # Make predictions
    predictions = model.predict(data)
    
    return predictions
