import sys

sys.path.append('/Users/maelysclerget/Desktop/ML/bio322_project/')

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from preprocessing import preprocessing_v1, apply_log_transformation, submission_file, calculate_feature_importance
import matplotlib.pyplot as plt

def linear_regression(feature_importance=False, apply_y_transformation=False):
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True)
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    
    if feature_importance:
        X_train, X_test = calculate_feature_importance(X_train, y_train, X_test)
        
    if apply_y_transformation:
        y_train = apply_log_transformation(y_train)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_train_pred)
    print('Training MSE:', mse)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print('CV MSE:', -cv_scores.mean())
    
    # Predict on test data
    y_test_pred = model.predict(X_test)
    
    # Create submission DataFrame
    submission = submission_file(y_test_pred)
    
    # Save submission to CSV
    submission.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/Submissions-files/sample_submission_LR.csv', index=False)
    print('Submission file saved successfully.')
    
        # Plot y_train vs y_train_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_pred, y_train, color='blue', label='Data points')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', label='y=x')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Linear Actual vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig('linear.png')
    plt.show()

    
def main():
    linear_regression()

if __name__ == '__main__':
    main()