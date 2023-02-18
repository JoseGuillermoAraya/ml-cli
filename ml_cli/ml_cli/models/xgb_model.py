import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

class TitanicXGBModel:
    
    def __init__(self, params):
        self.params = params
        self.model = None
        self.feature_names = None
        self.target_name = None
        self.feature_importance = None
        
    def fit(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.target_name = y_train.name
        self.feature_names = X_train.columns
        self.model = xgb.train(self.params, dtrain)
    
    def save(self, filename):
        joblib.dump(self, filename)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("XGBoost model is not trained yet")
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        return pd.Series(y_pred, name=self.target_name)

    def get_feature_importance(self):
        if self.model is not None and self.feature_names is not None:
            importance = self.model.get_score(importance_type='gain')
            importance = {self.feature_names[int(k[1:])] : v for k, v in importance.items()}
            self.feature_importance = importance
            return importance
        else:
            raise ValueError("The model has not been trained yet")