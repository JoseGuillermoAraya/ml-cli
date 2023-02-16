import argparse
from src.models.predict import predict
from src.models.train import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    # Predict subcommand
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('data_path', type=str, help='Path to data to make predictions on')
    predict_parser.add_argument('model_path', type=str, help='Path to the trained model')
    predict_parser.set_defaults(func=predict)

    # Train subcommand
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('data_path', type=str, help='Path to training data')
    train_parser.add_argument('model_path', type=str, help='Path to save the trained model')
    train_parser.add_argument('--params', type=str, default='{}', help='JSON string of XGB model parameters')
    train_parser.add_argument('--target', type=str, default='Survived', help='Name of target column')
    train_parser.set_defaults(func=train_model)

    args = parser.parse_args()
    args.func(args)
