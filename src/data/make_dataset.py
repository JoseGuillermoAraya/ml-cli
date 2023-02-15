import pandas as pd

def load_data(path):
    """Load data from path and return a pandas dataframe"""
    df = pd.read_csv(path)
    return df

def save_data(df, path):
    """Save data to path"""
    df.to_csv(path, index=False)

def split_data(df, target):
    """Split data into X and y"""
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y

def main():
    """Load data, split into X and y, save to data/processed"""
    df = load_data("data/raw/train.csv")
    X, y = split_data(df, "Survived")
    save_data(X, "data/processed/X.csv")
    save_data(y, "data/processed/y.csv")