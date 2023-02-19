import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from ml_cli.utils.metrics import evaluate, cross_validate


def test_evaluate():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 1, 0, 0])
    expected = {'accuracy': 0.6,
                'precision': 0.67,
                'recall': 0.67,
                'f1': 0.67,
                'auc': 0.58}
    assert evaluate(y_true, y_pred) == expected


def test_cross_validate():
    xgb_model = xgb.XGBClassifier()
    # Generate some fake data
    X, y = make_classification(n_samples=1000, random_state=42)

    # Test the function using the mock XGBoost model
    scores = cross_validate(X, y, xgb_model, cv=5, scoring='accuracy')
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 5
    assert np.all(scores >= 0) and np.all(scores <= 1)
