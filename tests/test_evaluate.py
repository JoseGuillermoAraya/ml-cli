import os
import pytest
import pandas as pd
from click.testing import CliRunner
from ml_cli.commands.evaluate import evaluate

@pytest.fixture
def output_file(tmp_path):
    return os.path.join(tmp_path, "results.csv")

@pytest.fixture
def log_file(tmp_path):
    return os.path.join(tmp_path, "test.log")

def test_evaluate(output_file, log_file):
    # Set up the input arguments
    input_file = './tests/data/titanic.csv'
    model_file = './tests/data/model.pkl'

    # Run the command-line interface using Click
    runner = CliRunner()
    result = runner.invoke(evaluate, ['--input-file', input_file, '--output-file', output_file, '--model-file', model_file, '--log-file', log_file])

    # Check that the command-line interface exits successfully
    assert result.exit_code == 0

    # Check that the evaluation results are saved to the output file
    assert os.path.exists(output_file)

    # Check that the output file contains the expected results
    results = pd.read_csv(output_file)
    assert results.shape == (1, 5)
    assert 'accuracy' in results.columns
    assert isinstance(results.loc[0, 'accuracy'], float)
