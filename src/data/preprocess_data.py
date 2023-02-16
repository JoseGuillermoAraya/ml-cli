import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

def drop_features(df, features):
    """Drop features from the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        features (list): List of features to drop.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with dropped features.
    """
    # Drop features
    df = df.drop(features, axis=1)
    
    return df

def impute_missing_values(df, strategy='mean'):
    """Impute missing values in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with imputed missing values.
    """
    # Impute missing values
    imputer = SimpleImputer(strategy=strategy)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return df

def normalize_data(df):
    """Normalize the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the normalized data.
    """
    # Normalize the data
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    return df

def encode_categorical_data(df):
    """Encode categorical data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with encoded categorical data.
    """
    # Encode categorical data
    encoder = OneHotEncoder(sparse=False)
    df = pd.DataFrame(encoder.fit_transform(df), columns=encoder.get_feature_names())
    
    return df

def create_title_feature(df):
    """Create title feature.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with title feature.
    """
    # Create title feature
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    
    return df

def feature_engineering(df):
    """Feature engineering.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with engineered features.
    """
    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df['IsAlone'].loc[df['FamilySize'] > 1] = 0
    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    df['FareBin'] = pd.qcut(df['Fare'], 4)
    df['AgeBin'] = pd.cut(df['Age'].astype(int), 5)
    
    return df