# ml_cli
This package contains a CLI to train and evaluate a ML model over the titanic problem data
Made using `poetry`

It has 3 commands:

## Train
`python cli.py train`
Arguments:
- `data_file`, path to the training data
- `model_file`, path to save the model to

Options:
- `'--log-file'`, default='train.log', Path to the log file
- `'--max_depth'`, default=3, Maximum depth of a tree
- `'--learning_rate'`, default=0.1, Learning rate
- `'--objective'`, default='binary:logistic', Objective function
- `'--eval_metric'`, default='logloss', Evaluation metric
- `'--min_child_weight'`, default=1, Minimum sum of instance weight (hessian) needed in a child
- `'--subsample'`, default=0.8, Subsample ratio of the training instances
- `'--colsample_bytree'`, default=0.8, Subsample ratio of columns when constructing each tree
- `'--min_child_weight'`, default=1, Minimum sum of instance weight (hessian) needed in a child
- `'--num_boost_round'`, default=100, Number of boosting rounds
- `'--early_stopping_rounds'`, default=10, Early stopping rounds
- `'--seed'`, default=42, Random seed
To train the model on some data

## Predict
`python cli.py predict`

Arguments:
- `input_file`, Path to the input CSV file
- `output_file`, Path to the output CSV file
- `model-file`, Path to the trained model

Options:
- `'--log-file'`, default='predict.log', Path to the log file
Predict target values from input data

## Evaluate
`python cli.py evaluate`

Arguments:
- `input-file`, Path to the input CSV file
- `model-file`, Path to the trained model
- `output-file`, Path to the output CSV file

Options:
- `'--log-file'`, default='evaluate.log', Path to the log file
Evaluate the model with accuracy, precission, f1, recall, auc

## Tests and coverage
`poetry run test`
Coverage will be shown in terminal and saved in html format to `./coverage-report`

## Building the package
`poetry build`
Package saves in `./dist`

## Installation
`poetry install`