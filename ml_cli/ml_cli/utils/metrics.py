from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb

def evaluate(y_true, y_pred):
    """
    Evaluate a binary classification model using common metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc}

def cross_validate(X, y, params, cv=5, scoring='accuracy'):
    """
    Cross validate a binary classification model using common metrics.
    """
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, dtrain, y, cv=cv, scoring=scoring)
    return scores