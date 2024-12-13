import sys

sys.path.append('/Users/maelysclerget/Desktop/ML/bio322_project/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from preprocessing import preprocessing_v1, apply_log_transformation, submission_file, scoring

def ridge_regression(apply_y_transformation=False):
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_scaling=False, apply_remove_outliers=False, 
                                                apply_correlation=False, apply_savgol=True)
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    if apply_y_transformation:
        y_train = apply_log_transformation(y_train)
        
    # Define the pipeline
    pipeline = Pipeline([
        ("polynomial", PolynomialFeatures()),
        ("regression", Ridge())
    ])
    
    # Define the parameter grid
    param_grid = {
        "polynomial__degree": np.arange(1, 3, 1),
        "regression__alpha": np.logspace(-12,-3,10)
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                               scoring='neg_mean_squared_error', 
                               return_train_score=True)
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print("Best accuracy:", grid_search.best_score_)
    print("Best parameters:", grid_search.best_params_)
    
    # Predict on training data
    y_train_pred = best_model.predict(X_train)
    if apply_y_transformation:
        y_train_pred = np.exp(y_train_pred)
        y_train_original = np.exp(y_train)
        mse = mean_squared_error(y_train_original, y_train_pred)
    else:
        mse = mean_squared_error(y_train, y_train_pred)
    print('Training MSE:', mse)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print('CV MSE:', -cv_scores.mean())
    
    # Predict on test data
    y_test_pred = best_model.predict(X_test)
    if apply_y_transformation:
        y_test_pred = np.exp(y_test_pred)
    
    # Create submission DataFrame
    submission = submission_file(y_test_pred)
    
    # Save submission to CSV
    submission.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/Submissions-files/sample_submission_RIDGE.csv', index=False)
    print('Submission file saved successfully.')
    
     # Calculate and print custom scoring
    score = scoring(y_train_pred, y_train)
    print('Score', score)
    
    # Plot y_train vs y_train_pred
    # Define range for parallel lines
    min_val = min(min(y_train), min(y_train_pred))
    max_val = max(max(y_train), max(y_train_pred))
    offset = 0.05 * (max_val - min_val)  # 5% of the range
    x = np.linspace(min_val, max_val, 100)

    # Plot y_train vs y_train_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_pred, y_train, color='blue', label='Data points', alpha=0.7)
    
    # Plot y = x line
    plt.plot(x, x, 'r--', label='y = x (Perfect Fit)', linewidth=2)
    
    # Plot parallel lines
    plt.plot(x, x + offset, 'g--', label=f'y = x + {offset:.2f} (+5%)', linewidth=2)
    plt.plot(x, x - offset, 'g--', label=f'y = x - {offset:.2f} (-5%)', linewidth=2)
    
    # Labels, title, legend
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Ridge Actual vs Predicted Values with ±5% Lines')
    plt.legend()
    plt.grid(True)
    
    # Save and display
    plt.savefig('Ridge.png')
    plt.show()


    
    
def main():
    ridge_regression(apply_y_transformation=False)
    
if __name__ == '__main__':
    main()