
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import chi2
from preprocessing import preprocessing_v1, submission_file


def best_param_RF():
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'random_state': [42]
    }

    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Preprocess the data
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_scaling=True, apply_remove_outliers=True)
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_rf = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)
    return best_rf

def random_forest_linear_regression(apply_y_transformation=False):
    
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_correlation=False, apply_scaling=True, apply_remove_outliers=True, apply_variance_threshold=False, apply_random_forest=True)
    
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    if apply_y_transformation:
        y_train = np.log1p(y_train)  # Apply log transformation
        
    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    
    # Fit the model
    rf.fit(X_train, y_train)
    
    # Predict on training data
    y_train_pred = rf.predict(X_train)
    if apply_y_transformation:
        y_train_pred = np.exp(y_train_pred)  # Apply inverse log transformation
        y_train_original = np.exp(y_train)
        mse = mean_squared_error(y_train_original, y_train_pred)
    else:
        mse = mean_squared_error(y_train, y_train_pred)
    print('Training MSE:', mse)
    
    # Cross-validation score
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print('CV MSE:', -cv_scores.mean())
    
    # Predict on test data
    y_test_pred = rf.predict(X_test)
    if apply_y_transformation:
        y_test_pred = np.exp(y_test_pred)  # Apply inverse log transformation
    
    # Create submission DataFrame
    submission = submission_file(y_test_pred)
    
    # Save submission to CSV
    submission.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/sample_submission_RF.csv', index=False)
    print('Submission file saved successfully.')
    