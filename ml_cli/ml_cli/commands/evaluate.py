import click
import joblib
import pandas as pd
from ml_cli.utils.logger import get_logger
from ml_cli.data.make_dataset import make_dataset
from ml_cli.data.preprocess_data import preprocess_data

# Define the command-line arguments
@click.command()
@click.option('--test-file', type=click.Path(exists=True), required=True, help='Path to the test CSV file')
@click.option('--output-file', type=click.Path(), default='results.csv', help='Path to the output CSV file')
@click.option('--model-file', default='model.bin', help='Path to the trained model')
@click.option('--log-file', default='predict.log', help='Path to the log file')

def evaluate(test_file, output_file, model_file, log_file):
    logger = get_logger(__name__, log_file)
    # Load the XGBoost model and feature encoder
    logger.info(f'Loading model from {model_file}...')
    model = joblib.load(model_file)

    # Load the test data into a Pandas DataFrame
    logger.info(f'Loading data from {test_file}...')
    X_test, y_test = make_dataset(test_file, 'Survived')

    #Preprocess the data
    logger.info('Preprocessing data...')
    X_test = preprocess_data(X_test)

    # Evaluate model
    logger.info('Evaluating model...')
    accuracy = model.evaluate(X_test, y_test)
    cross_validation = model.cross_validate(X_test, y_test)


    # Save the performance metrics to a CSV file
    logger.info(f'Saving evaluation results to {output_file}...')
    results = pd.DataFrame({'accuracy': [accuracy], 'cross_validation_score': [cross_validation]})
    results.to_csv(output_file, index=False)

    # Log success message
    logger.info(f'Successfully saved evaluation results to {output_file}')

if __name__ == '__main__':
    evaluate()
