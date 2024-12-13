import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import OrthogonalMatchingPursuit
from preprocessing import preprocessing_v1, submission_file
import matplotlib.pyplot as plt

def orthogonal_matching_pursuit():
    
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_correlation=False, apply_remove_outliers=False, apply_savgol=True)
    
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    # Define the pipeline
    pipeline = Pipeline([
        ("polynomial", PolynomialFeatures()),
        ("regression", OrthogonalMatchingPursuit())
    ])
    
    # Define the parameter grid
    param_grid = {
        "polynomial__degree": np.arange(1, 3, 1),
        "regression__n_nonzero_coefs": range(1, 100, 10)
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
    submission.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/sample_submission_OMP.csv', index=False)
    print('Submission file saved successfully.')
    
    # Plot y_train vs y_train_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_pred, y_train, color='blue', label='Data points')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='y=x')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('OMP Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('OMP.png')
    plt.show()

def main():
    orthogonal_matching_pursuit()   
    
if __name__ == '__main__':  
    main()