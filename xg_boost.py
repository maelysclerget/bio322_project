from xgboost import XGBRegressor
from preprocessing import preprocessing_v1, submission_file
import numpy as np

def xg_boost():
    
    X_train, X_test, y_train = preprocessing_v1(apply_one_hot=True, apply_scaling=True)
    X_train = X_train.drop(columns=['sample_name'])
    X_test = X_test.drop(columns=['sample_name'])
    target_column = 'PURITY'


    param_grid = {
        'colsample_bytree': np.linspace(0.5, 1, 5),
        'subsample': np.linspace(0.5, 1, 5),
        'max_depth': np.arange(2, 7, 1)
    }


    model = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Create submission DataFrame
    submission = submission_file(y_pred)
    
    # Save submission to CSV
    submission.to_csv('/Users/alicepriolet/Desktop/ML/epfl-bio-322-2024/sample_submission_LR.csv', index=False)
    print('Submission file saved successfully.')


def main():
    xg_boost()
    

if __name__ == '__main__':
    main() 