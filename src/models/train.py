import joblib
from src.models.model import TitanicXGBModel

def train_model(X_train, y_train, model_path, model_params):
    # Instantiate the model with the provided parameters
    model = TitanicXGBModel(model_params)

    # Train the model
    model.fit(X_train, y_train)

    # Save the model to disk
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")