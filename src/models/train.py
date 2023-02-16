import joblib
from src.models.model import TitanicXGBModel

def train_model(X_train, y_train, model_name, model_params):
    # Instantiate the model with the provided parameters
    model = TitanicXGBModel(model_params)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model to disk
    joblib.dump(model, f"../../data/models/{model_name}")

    print(f"Model saved to ./data/models/{model_name}")