import joblib
import pandas as pd
from src.models.model import TitanicXGBModel
from src.data.make_dataset import make_dataset
from src.data.preprocess_data import preprocess_data

def train_model(data_path, model_path, model_params, target):
    # Instantiate the model with the provided parameters
    model = TitanicXGBModel(model_params)

    # Load the training data
    data = pd.read_csv(data_path)

    # Make data set
    X_train, y_train = make_dataset(data)

    # Preprocess the data
    X_train = preprocess_data(X_train)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model to disk
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")