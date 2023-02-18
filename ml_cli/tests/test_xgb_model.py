from ml_cli.models.xgb_model import TitanicXGBModel
from sklearn.datasets import make_classification
import pandas as pd
import os 
import pytest

def test_xgb_instanciation():
    """Test that the model can be instantiated without errors"""
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    xgb_model = TitanicXGBModel(params)
    assert xgb_model.params == params
    assert xgb_model.model is None
    assert xgb_model.feature_names is None
    assert xgb_model.target_name is None
    assert xgb_model.feature_importance is None
    assert isinstance(xgb_model, TitanicXGBModel)

def test_xgb_fit():
    """Test that the model can be fit to some training data"""
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    xgb_model = TitanicXGBModel(params)

    # Create the data set
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Assign column names to X
    X = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10'])

    # Assign a name to the target column
    y_name = 'target'

    # Assign column names to y
    y = pd.Series(y, name=y_name)

    xgb_model.fit(X, y)
    assert xgb_model.model is not None
    assert xgb_model.feature_names is not None
    assert xgb_model.target_name is not None

def test_xgb_save():
    """Test that the model can be saved to a file"""
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    xgb_model = TitanicXGBModel(params)

    # Create the data set
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Assign column names to X
    X = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10'])

    # Assign a name to the target column
    y_name = 'target'

    # Assign column names to y
    y = pd.Series(y, name=y_name)

    xgb_model.fit(X, y)
    xgb_model.save('test_xgb_model.pkl')
    assert os.path.exists('test_xgb_model.pkl')
    os.remove('test_xgb_model.pkl')

def test_xgb_predict():
    """Test that the model can make predictions on new data"""
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    xgb_model = TitanicXGBModel(params)

    # Create the data set
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Assign column names to X
    X = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10'])

    # Assign a name to the target column
    y_name = 'target'

    # Assign column names to y
    y = pd.Series(y, name=y_name)

    xgb_model.fit(X, y)
    y_pred = xgb_model.predict(X)
    assert y_pred is not None
    assert isinstance(y_pred, pd.Series)
    assert y_pred.shape[0] == X.shape[0]

def test_predict_untrained_model():
    """Test that the model raises an error when trying to make predictions on untrained model"""
    with pytest.raises(ValueError, match="XGBoost model is not trained yet"):
        params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
        model = TitanicXGBModel(params)
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        model.predict(X_test)

def test_xgb_get_feature_importance():
    """Test that the model can get the feature importance"""
    params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
    xgb_model = TitanicXGBModel(params)

    # Create the data set
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Assign column names to X
    X = pd.DataFrame(X, columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10'])

    # Assign a name to the target column
    y_name = 'target'

    # Assign column names to y
    y = pd.Series(y, name=y_name)

    xgb_model.fit(X, y)
    feature_importance = xgb_model.get_feature_importance()
    assert feature_importance is not None
    assert isinstance(feature_importance, pd.Series)
    assert feature_importance.shape[0] == X.shape[1]

def test_xgb_get_feature_importance_untrained_model():
    """Test that the model raises an error when trying to get feature importance on untrained model"""
    with pytest.raises(ValueError, match="XGBoost model is not trained yet"):
        params = {'objective': 'binary:logistic', 'eval_metric': 'logloss'}
        model = TitanicXGBModel(params)
        model.get_feature_importance()