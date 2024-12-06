
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge
from bayes_opt import BayesianOptimization
from preprocessing import preprocessing_v1, apply_log_transformation, submission_file

def bayesian_ridge_regression(apply_y_transformation=False):
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_correlation=False, apply_scaling=True, apply_remove_outliers=False, apply_random_forest=True)
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    if apply_y_transformation:
        y_train = apply_log_transformation(y_train)
    
    # Define the objective function for Bayesian Optimization
    def objective(alpha_1, alpha_2, lambda_1, lambda_2):
        model = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        return -cv_scores.mean()
    
    # Define the parameter bounds
    params_bayesian_ridge = {
        'alpha_1': (1e-6, 1e-3),
        'alpha_2': (1e-6, 1e-3),
        'lambda_1': (1e-6, 1e-3),
        'lambda_2': (1e-6, 1e-3)
    }
    
    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(f=objective, pbounds=params_bayesian_ridge, random_state=42)
    
    # Maximize the objective function
    optimizer.maximize(init_points=10, n_iter=50)
    
    # Get the best parameters
    best_params = optimizer.max['params']
    print("Best parameters:", best_params)
    
    # Train the final model with the best parameters
    bayesian_ridge = BayesianRidge(**best_params)
    bayesian_ridge.fit(X_train, y_train)
    
    # Predict on training data
    y_train_pred = bayesian_ridge.predict(X_train)
    if apply_y_transformation:
        y_train_pred = np.expm1(y_train_pred)
        y_train_original = np.expm1(y_train)
        mse = mean_squared_error(y_train_original, y_train_pred)
    else:
        mse = mean_squared_error(y_train, y_train_pred)
    print('Training MSE:', mse)
    
    # Cross-validation score
    cv_scores = cross_val_score(bayesian_ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print('CV MSE:', -cv_scores.mean())
    
    # Predict on test data
    y_test_pred = bayesian_ridge.predict(X_test)
    if apply_y_transformation:
        y_test_pred = np.expm1(y_test_pred)
    
    # Create submission DataFrame
    submission = submission_file(y_test_pred)
    
    # Save submission to CSV
    submission.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/sample_submission_BAYESIAN_RIDGE.csv', index=False)
    print('Submission file saved successfully.')
