import sys

sys.path.append('/Users/maelysclerget/Desktop/ML/bio322_project/')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import preprocessing_v1, apply_log_transformation
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, OrthogonalMatchingPursuit, ElasticNet


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
    plt.savefig('Plots/variance.png')
    plt.show()

    
#plot avec et sans log transformation    
def plot_response_variable(apply_y_transformation=True, output_filename="Plots/response_variable.png", xlim=None):
    
    # Get preprocessed data
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_scaling=True, apply_remove_outliers=False)

    # Apply log transformation if specified
    if apply_y_transformation:
        y_train = apply_log_transformation(y_train)
        title_suffix = " with log transformation"
    else:
        title_suffix = ""

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(y_train, kde=True)
    plt.title(f'Distribution of Response Variable (PURITY){title_suffix}')
    plt.xlabel('PURITY')
    plt.ylabel('Frequency')
    
    # Set x-axis limits if specified
    if xlim:
        plt.xlim(xlim)
    
    # Save the plot
    plt.savefig(output_filename)
    print(f"Plot saved as {output_filename}")
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
def calculate_cv_mse(model, X_train, y_train):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -cv_scores.mean()

def plot_cv_mse_results():
    # Get preprocessed data
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_correlation=True, apply_savgol=True)
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    models = {
        #'Linear': LinearRegression(),
        #'Polynomial': make_pipeline(PolynomialFeatures(degree=1), LinearRegression()),
        'Ridge': Ridge(),
        'Lasso': Lasso(), 
        'OMP': OrthogonalMatchingPursuit(), 
        'ElasticNet': ElasticNet() 
    }
    
    cv_mse_results = []
    
    for name, model in models.items():
        cv_mse = calculate_cv_mse(model, X_train, y_train)
        cv_mse_results.append((name, cv_mse))
        print(f'{name} - CV MSE: {cv_mse:.4f}')
    
    # Plot the CV MSE results
    model_names, mse_values = zip(*cv_mse_results)
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, mse_values, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('CV MSE')
    plt.title('Cross-Validation MSE for Different Models')
    plt.grid(True)
    plt.savefig('Plots/cv_mse_results.png')
    plt.show()

    
def main():
    #plot_variance()
    #plot_response_variable()
    #plot_ridge_regression_data(apply_y_transformation=False, alpha=0.001, degree=1)
    plot_cv_mse_results()

    
if __name__ == '__main__':
    main()
    