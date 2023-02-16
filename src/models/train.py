from src.data import make_dataset
from src.data import preprocess_data
from src.models.model import TitanicXGBModel

def train_model():
    # Load the data
    X, y = make_dataset()

    # Preprocess the data
    X = preprocess_data(X)

    # Create the model
    model = TitanicXGBModel()

    # Train the model
    model.fit(X, y)

    return model