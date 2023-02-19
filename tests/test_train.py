import os
import pytest
from click.testing import CliRunner
from ml_cli.commands.train import train

@pytest.fixture
def input_data_file(tmp_path):
    input_data = tmp_path / "input_data.csv"
    with open(input_data, "w") as f:
        f.write("PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked\n1,0,3,\"Braund, Mr. Owen Harris\",male,22,1,0,A/5 21171,7.25,,S\n2,1,1,\"Cumings, Mrs. John Bradley (Florence Briggs Thayer)\",female,38,1,0,PC 17599,71.2833,C85,C\n")
    return input_data

@pytest.fixture
def output_model_file(tmp_path):
    return os.path.join(tmp_path, "model.pkl")

def test_train(input_data_file, output_model_file):
    runner = CliRunner()
    result = runner.invoke(train, [
        "--data-file", str(input_data_file),
        "--log-file", "test.log",
        "--model-file", str(output_model_file),
        "--max_depth", "5",
        "--learning_rate", "0.05",
        "--objective", "binary:logistic",
        "--eval_metric", "logloss",
        "--min_child_weight", "1",
        "--subsample", "0.8",
        "--colsample_bytree", "0.8",
        "--num_boost_round", "10",
        "--early_stopping_rounds", "5",
        "--seed", "42"
    ])
    assert result.exit_code == 0
    assert os.path.exists(output_model_file)
