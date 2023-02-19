import click
import pandas as pd
import joblib
from ml_cli.utils.logger import get_logger

@click.command()
@click.option('--model-file', type=click.Path(exists=True), required=True, help='Path to the trained model')
@click.option('--output-file', type=click.Path(), required=True, help='Path to the output file')

def get_feature_importance(model_file, output_file):
    """Get the feature importances from the trained model."""
    logger = get_logger(__name__)

    # Load the trained model
    logger.info(f'Loading model from {model_file}...')
    model = joblib.load(model_file)

    # Get the feature importances
    logger.info('Getting feature importances...')
    feature_importances = model.get_feature_importance()

    # Save the feature importances to a CSV file
    logger.info(f'Saving feature importances to {output_file}...')
    feature_importances_df = pd.DataFrame(feature_importances)
    feature_importances_df.index.name = 'feature'
    feature_importances_df.to_csv(output_file, index=True)

    click.echo(f'Successfully saved feature importances to {output_file}.')