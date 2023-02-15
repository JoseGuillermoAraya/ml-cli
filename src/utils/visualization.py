import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(df, title):
    """Plot correlation matrix in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        title (str): Title of the plot.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with correlation matrix.
    """
    # Plot correlation matrix
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', ax=ax)
    ax.set_title(title)
    plt.show()
    return df

def plot_count_plot(df, column, hue, title):
    """Plot count plot in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        column (str): Column to plot.
        hue (str): Column to split the plot.
        title (str): Title of the plot.
        
    Returns:
        df (pandas.DataFrame): Dataframe containing the data with count plot.
    """
    # Plot count plot
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.countplot(x=column, hue=hue, data=df, ax=ax)
    ax.set_title(title)
    plt.show()
    return df