import pytest
from ml_cli.data.make_dataset import *

def test_load_data():
    """Test load_data function"""
    # Load data
    df = load_data('./tests/data/titanic.csv')
    
    # Test output
    assert df.shape == (891, 12)

def test_split_data():
    """Test split_data function"""
    # Load data
    df = load_data('./tests/data/titanic.csv')
    
    # Split data
    X, y = split_data(df, 'Survived')
    
    # Test output
    assert X.shape == (891, 11)
    assert y.shape == (891,)

def test_make_dataset():
    """Test make_dataset function"""
    # Load data
    X, y = make_dataset('./tests/data/titanic.csv', 'Survived')
    
    # Test output
    assert X.shape == (891, 11)
    assert y.shape == (891,)