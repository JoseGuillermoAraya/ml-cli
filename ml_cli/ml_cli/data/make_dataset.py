import pandas as pd

def load_data(path):
    """Load data from path and return a pandas dataframe"""
    df = pd.read_csv(path)
    return df

def split_data(df, target):
    """Split data into X and y"""
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

def make_dataset(data_path, target):
    """Make dataset from data path and target"""
    # Load data
    df = load_data(data_path)
    
    # Split data
    X, y = split_data(df, target)
    
    return X, y