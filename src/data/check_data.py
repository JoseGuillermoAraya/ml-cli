import pandas as pd

def check_missing_values(df):
    """Check for missing values in the data. Returns a dataframe with
    the amount of missing values in each column and the percentage of missing values
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with missing values.
    """
    # Check for missing values
    df = df.isnull().sum()
    df = df[df > 0]
    df = df.sort_values(ascending=False)
    df = pd.DataFrame(df, columns=['Missing Values'])
    df['% Missing Values'] = df['Missing Values'] / len(df) * 100
    return df

def check_data_types(df):
    """Check data types in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with data types.
    """
    # Check data types
    df = df.dtypes
    return df

def check_for_duplicates(df):
    """Check for duplicates in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with duplicates.
    """
    # Check for duplicates
    df = df.duplicated().sum()
    return df

def check_data_distribution(df):
    """Check data distribution in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with data distribution.
    """
    # Check data distribution
    df = df.describe()
    return df

def check_column_distribution(df):
    """Check column distribution, skewness, and kurtosis in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with column distribution.
    """
    # Check column distribution, skewness, and kurtosis
    skew = df.skew()
    kurtosis = df.kurtosis()
    return skew, kurtosis

def check_for_outliers(df):
    """Check for outliers in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with outliers.
    """
    # Check for outliers
    df = df.boxplot()
    return df