from sklearn.model_selection import train_test_split

def split_train_test(df, test_size=0.2, random_state=42):
    """Split data into train and test sets.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random state.
        
    Returns:
        df_train (pandas.DataFrame): Dataframe containing the training data.
        df_test (pandas.DataFrame): Dataframe containing the testing data.
    """
    # Split data into train and test sets
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return df_train, df_test

def split_train_validation_test(df, validation_size=0.2, test_size=0.2, random_state=42):
    """Split data into train, validation, and test sets.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        validation_size (float): Fraction of data to use for validation.
        test_size (float): Fraction of data to use for testing.
        random_state (int): Random state.
        
    Returns:
        df_train (pandas.DataFrame): Dataframe containing the training data.
        df_validation (pandas.DataFrame): Dataframe containing the validation data.
        df_test (pandas.DataFrame): Dataframe containing the testing data.
    """
    # Split data into train and test sets
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Split data into train and validation sets
    df_train, df_validation = train_test_split(df_train, test_size=validation_size, random_state=random_state)
    
    return df_train, df_validation, df_test