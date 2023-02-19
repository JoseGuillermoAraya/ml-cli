import os
import pandas as pd
from click.testing import CliRunner
from ml_cli.commands.get_feature_importance import get_feature_importance

def test_get_feature_importance(tmpdir):
    # Call the get_feature_importance function on the sample input file
    runner = CliRunner()
    output_file = str(tmpdir.join('feature_importances.csv'))
    result = runner.invoke(get_feature_importance, ['--model-file', './tests/data/model.pkl', '--output-file', output_file])

    # Check that the command was successful
    assert result.exit_code == 0

    # Check that the output file was created and contains expected number of predictions
    output_data = pd.read_csv(output_file)
    assert len(output_data) == 17
    assert 'feature' in output_data.columns
    assert 'importance' in output_data.columns

    # Clean up the output file
    os.remove(output_file)