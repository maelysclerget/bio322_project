
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import HuberRegressor
from preprocessing import preprocessing_v1, apply_log_transformation, submission_file, calculate_feature_importance


def huber_regression(apply_feature_importance=True, apply_y_transformation=True):
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_correlation=False, apply_scaling=True, apply_remove_outliers=True, apply_variance_threshold=False, apply_random_forest=True)
    
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    if apply_feature_importance:
        X_train, X_test = calculate_feature_importance(X_train, y_train, X_test)
    
    if apply_y_transformation:
        y_train = apply_log_transformation(y_train)
        
    # Define the parameter grid for epsilon
    param_grid = {'epsilon': np.arange(1, 2, 0.5)}
     
    # Initialize the HuberRegressor
    huber = HuberRegressor()
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=huber, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_huber = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    
    # Predict on training data
    y_train_pred = best_huber.predict(X_train)
    if apply_y_transformation:
        y_train_pred = np.exp(y_train_pred)
        y_train_original = np.exp(y_train)
        mse = mean_squared_error(y_train_original, y_train_pred)
    else:
        mse = mean_squared_error(y_train, y_train_pred)
    print('Training MSE:', mse)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_huber, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print('CV MSE:', -cv_scores.mean())
    
    # Predict on test data
    y_test_pred = best_huber.predict(X_test)
    if apply_y_transformation:
        y_test_pred = np.exp(y_test_pred)
    
    # Create submission DataFrame
    submission = submission_file(y_test_pred)
    
    # Save submission to CSV
    submission.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/sample_submission_HUBER.csv', index=False)
    print('Submission file saved successfully.')