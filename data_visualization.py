
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import preprocessing_v1, apply_log_transformation


#MAELYS:
#Quel trehsold de variance choisir pour selectionner les features
def plot_variance():
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_scaling=True, apply_remove_outliers=True)
    
    # Select only numerical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    
    # Calculate variance for each numerical feature
    variance = X_train[numerical_cols].var()
    
    # Plot variance for each feature
    plt.figure(figsize=(10, 6))
    sns.barplot(x=variance, y=variance.index)
    plt.title('Variance of Features')
    plt.xlabel('Variance')
    plt.ylabel('Feature')
    plt.show()
    plt.savefig('/Users/maelysclerget/Desktop/ML/bio322_project/plots/variance.png')
    
#plot avec et sans log transformation    
def plot_response_variable(apply_y_transformation=True):
    
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_scaling=True, apply_remove_outliers=False)

    if apply_y_transformation:
        y_train = apply_log_transformation(y_train)
        
    plt.figure(figsize=(10, 6))
    sns.histplot(y_train, kde=True)
    plt.title('Distribution of Response Variable (PURITY)')
    plt.xlabel('PURITY')
    plt.ylabel('Frequency')
    #plt.xlim(0, 10)
    plt.savefig('/Users/maelysclerget/Desktop/ML/bio322_project/plots/response_variable_log.png')
    plt.show()


def plot_boxplot(title, ax=None):
    """
    Function to calculate summary statistics and plot a boxplot for numeric columns in a DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the dataset.
    - col_name: string, the column name of the feature to inspect.
    - title: string, the title of the plot.
    - ax: matplotlib axis object, allows the plot to be part of a larger figure.
    """
    
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_scaling=True, apply_remove_outliers=True)

    # Drop missing values
    non_nan_series = y_train.dropna()
    
    # Calculate summary statistics
    print(y_train.describe())
    
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(y_train):
        
        # Plot boxplot for numeric data
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(non_nan_series, vert=False)
        ax.set_title(title)
        ax.set_xlabel('PURITY')
        ax.grid(True)
        
        # Show plot if standalone
        if ax is None:
            plt.show()
    else:
        print(f'PURITY is not a numeric column. Skipping boxplot.\n')
        
#Ne pas oublier le mettre le plot de correlation matrix dans preprocessing.py 
""" def print_mse_results():
    # Define the paths to the files containing the MSE results
    mse_files = {
        'ElasticNet': '/Users/maelysclerget/Desktop/ML/bio322_project/elasticnet_regression.py',
        'Ridge': '/Users/maelysclerget/Desktop/ML/bio322_project/ridge_regression.py',
        'Lasso': '/Users/maelysclerget/Desktop/ML/bio322_project/lasso_regression.py',
        'Huber': '/Users/maelysclerget/Desktop/ML/bio322_project/huber_regression.py',
        'OMP': '/Users/maelysclerget/Desktop/ML/bio322_project/OMP.py',
        'Linear': '/Users/maelysclerget/Desktop/ML/bio322_project/linear_regression.py',
        'Polynomial': '/Users/maelysclerget/Desktop/ML/bio322_project/polynomial_regression.py',
        'BayesianRidge': '/Users/maelysclerget/Desktop/ML/bio322_project/bayesian_ridge_regression.py',
        'RandomForest': '/Users/maelysclerget/Desktop/ML/bio322_project/RF_regression.py'
    }
    
    # Read and print the MSE results from each file
    for model_name, file_path in mse_files.items():
        mse_data = pd.read_csv(file_path)
        train_mse = mse_data['train_mse'].values[0]
        test_mse = mse_data['test_mse'].values[0]
        print(f'{model_name} - Training MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}')

if __name__ == '__main__':
    print_mse_results() """
    
def main():
    plot_variance()
    plot_response_variable()
    plot_boxplot()
    
if __name__ == '__main__':
    main()
    