import pandas as pd
import os

from click.testing import CliRunner
from ml_cli.commands.predict import predict


def test_predict(tmpdir):
    sample_input_file = './tests/data/titanic_without_target.csv'
    trained_model = './tests/data/model.pkl'
    # Call the predict function on the sample input file
    runner = CliRunner()
    output_file = str(tmpdir.join('predictions.csv'))
    result = runner.invoke(predict, ['--input_file', sample_input_file, '--model-file', trained_model, '--output_file', output_file])

    # Check that the command was successful
    assert result.exit_code == 0

    # Check that the output file was created and contains expected number of predictions
    output_data = pd.read_csv(output_file)
    assert len(output_data) == 183
    assert 'predictions' in output_data.columns
    
    # check that predictions are binary
    assert set(output_data['predictions'].unique()) == {0, 1}

    # Clean up the output file
    os.remove(output_file)
