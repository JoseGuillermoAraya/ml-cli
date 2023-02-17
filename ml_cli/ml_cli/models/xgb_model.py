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
        
    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        y_pred = self.model.predict(dtest)
        return y_pred
        
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred.round())
        print(f"Accuracy: {accuracy:.2f}")
    
    def cross_validate(self, X, y, cv=5, scoring='accuracy'):
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        return scores.mean()

    def get_feature_importance(self):
        if self.model is not None and self.feature_names is not None:
            importance = self.model.get_score(importance_type='gain')
            importance = {self.feature_names[int(k[1:])] : v for k, v in importance.items()}
            self.feature_importance = importance
            return importance
        else:
            raise ValueError("The model has not been trained yet")