# ml_cli
This package contains a CLI to train and evaluate a ML model over the titanic problem data

It has 3 commands:

## Train
`python cli.py train --data-file=/path/to/input/data`
To train the model on some data

## Predict
`python cli.py predict --input-file=/path/to/input/data`
Predict taregt values from input data
## Evaluate
`python cli.py evaluate --input-file=/path/to/input/data --model-file=/path/to/model/file`
Evaluate the model with accuracy, precission, f1, recall, auc
