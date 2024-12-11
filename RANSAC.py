import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from scipy.stats import chi2
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from preprocessing import preprocessing_v1, submission_file
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def RANSAC_Regressor(): 
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_correlation=True, apply_scaling=False, 
                                                apply_remove_outliers=False, apply_variance_threshold=False, apply_random_forest=True, 
                                                apply_robust_scaling=False)
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    model = RANSACRegressor()
    model.fit(X_train, y_train)
    
    # Define the pipeline
    pipeline = Pipeline([
        ('ransac', RANSACRegressor())
    ])
    
    param_grid = {
    'ransac__residual_threshold': [1.0, 2.0, 5.0],
    'ransac__min_samples': [0.5, 0.6, 0.7]
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
    mse = mean_squared_error(y_train, y_train_pred)
    print('Training MSE:', mse)
    
    # Cross-validation score
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print('CV MSE:', -cv_scores.mean())
    
    # Predict on test data
    y_test_pred = best_model.predict(X_test)
    
    # Create submission DataFrame
    submission = submission_file(y_test_pred)
    
    # Save submission to CSV
    submission.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/sample_submission_RANSAC.csv', index=False)
    print('Submission file saved successfully.')


def main():
    RANSAC_Regressor()
    
if __name__ == '__main__':
    main()