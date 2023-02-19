from ml_cli.data.preprocess_data import *
from ml_cli.data.make_dataset import load_data
import numpy as np

def test_drop_features():
    """Test drop_features function"""
    # Load data
    df = load_data('./tests/data/titanic.csv')
    
    # Drop features
    df = drop_features(df, ['Name', 'Ticket', 'Cabin'])
    
    # Test output
    assert df.shape == (891, 9)
    assert 'Name' not in df.columns
    assert 'Ticket' not in df.columns
    assert 'Cabin' not in df.columns

def test_create_title_feature():
    """Test create_title_feature function"""
    # Load data
    df = load_data('./tests/data/titanic.csv')
    initial_shape = df.shape
    
    # Create title feature
    df = create_title_feature(df)
    
    # Test output
    assert df.shape == (initial_shape[0], initial_shape[1] + 1)
    assert 'Title' in df.columns
    assert df['Title'].nunique() == 17
    assert all([df['Name'][i].find(df['Title'][i]) != -1 for i in range(len(df))])

def test_impute_missing_values():
    """Test impute_missing_values function"""
    # Load data
    df = load_data('./tests/data/titanic.csv')
    
    #Check for null values on age
    assert df['Age'].isnull().sum() > 0

    # Get most frequent age value
    most_frequent_age = df['Age'].value_counts().index[0]

    #get passengerIds of passengers with missing age
    missing_age_passengerIds = df[df['Age'].isnull()]['PassengerId']

    # Impute missing values
    df = impute_missing_values(df, 'Age', 'most_frequent')
    
    # Test that there are no null values
    assert df['Age'].isnull().sum() == 0

    # Test that the most frequent age value is imputed
    assert all([df[df['PassengerId'] == passengerId]['Age'].values[0] == most_frequent_age for passengerId in missing_age_passengerIds])

def test_create_band_feature():
    """Test create_band_feature function"""
    # Load data
    df = load_data('./tests/data/titanic_no_missing_values.csv')
    initial_shape = df.shape
    
    # Create band feature
    df = create_band_feature(df, 'Age', 'AgeBand', 10)

    # get max and min ages
    max_age = int(df['Age'].max())
    min_age = int(df['Age'].min())
    bin_size = int((max_age - min_age) / 10)

    # Test output
    assert df.shape == (initial_shape[0], initial_shape[1] + 1)
    assert 'AgeBand' in df.columns
    assert df['AgeBand'].nunique() == 10
    assert all([int(df['Age'][i]) >= df['AgeBand'][i] * bin_size and int(df['Age'][i]) <= (df['AgeBand'][i] * bin_size) + bin_size for i in range(len(df))])

def test_create_feature_sum():
    """Test create_feature_sum function"""
    # Load data
    df = load_data('./tests/data/titanic_no_missing_values.csv')
    initial_shape = df.shape
    
    # Create feature sum
    df = create_feature_sum(df, ['SibSp', 'Parch'], 'FamilySize', 1)

    # Test output
    assert df.shape == (initial_shape[0], initial_shape[1] + 1)
    assert 'FamilySize' in df.columns
    assert all([df['FamilySize'][i] == df['SibSp'][i] + df['Parch'][i] + 1 for i in range(len(df))])
    
def test_create_feature_from_column():
    """Test create_feature_from_column function"""
    # Load data
    df = load_data('./tests/data/titanic_no_missing_values.csv')
    initial_shape = df.shape
    
    # Create feature from column
    df = create_feature_from_column(df, "Age", "Result", lambda x: x > 30)

    # Test output
    assert df.shape == (initial_shape[0], initial_shape[1] + 1)
    assert 'Result' in df.columns
    assert all([df['Result'][i] == (df['Age'][i] > 30) for i in range(len(df))])

def test_encode_categorical_features():
    """Test encode_categorical_features function"""
    #Load data
    df = load_data('./tests/data/titanic_no_missing_values.csv')
    initial_shape = df.shape
    features_to_encode = ['Sex', 'Embarked']

    # sum of Unique values for each feature
    sum_unique_values = sum([df[feature].nunique() for feature in features_to_encode])

    # Encode categorical features
    df = encode_categorical_features(df, features_to_encode)

    # Test output
    assert df.shape == (initial_shape[0], initial_shape[1] + sum_unique_values - len(features_to_encode))
    assert all([feature not in df.columns for feature in features_to_encode])

def test_preprocess_data():
    """Test preprocess_data function"""
    df = load_data('./tests/data/titanic.csv')
    initial_shape = df.shape
    X = preprocess_data(df)

    print(X.columns)
    # Test output
    assert X.shape == (initial_shape[0], initial_shape[1] + 6)
    assert 'Name' not in X.columns
    assert 'Ticket' not in X.columns
    assert 'Cabin' not in X.columns
    assert 'Age' not in X.columns
    assert 'PassengerId' not in X.columns
    assert 'AgeBand' in X.columns
    assert 'FamilySize' in X.columns
    assert 'Sex_female' in X.columns
    assert 'Sex_male' in X.columns
    assert 'Embarked_C' in X.columns
    assert 'Embarked_Q' in X.columns
    assert 'Embarked_S' in X.columns
    assert 'Title_Master' in X.columns
    assert 'Title_Miss' in X.columns
    assert 'Title_Mr' in X.columns
    assert 'Title_Mrs' in X.columns
    assert 'Title_Other' in X.columns

