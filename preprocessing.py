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
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.signal import savgol_filter
from scipy.stats import zscore

    
def preprocessing_v1(apply_one_hot=False, apply_scaling=False, apply_pca=False, 
                     apply_correlation=False, apply_remove_outliers=False, 
                     apply_variance_threshold=False, apply_random_forest=False, 
                     apply_robust_scaling=False, apply_savgol=False):   
    train_data_og = pd.read_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/train.csv')
    test_data_og = pd.read_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/test.csv')
   
    train_data = train_data_og.copy()
    test_data = test_data_og.copy()
    
    train_data = train_data.drop(columns=['prod_substance'])
    test_data = test_data.drop(columns=['prod_substance'])
    
    non_wavelength_cols = ['device_serial', 'substance_form_display', 'measure_type_display']
    wavelength_cols = train_data.columns[5:]
    
    # Remove NaN values
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    
    if apply_one_hot:
        # One Hot encoding 
        encoder = OneHotEncoder(drop='first',sparse_output=False, handle_unknown='ignore')
        X_train_encoded = encoder.fit_transform(train_data[non_wavelength_cols])
        X_test_encoded = encoder.transform(test_data[non_wavelength_cols])
        
        # Convert encoded features to DataFrame
        X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(non_wavelength_cols))
        X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(non_wavelength_cols))
        
        train_data_combined = pd.concat([pd.DataFrame(X_train_encoded_df), train_data[wavelength_cols].reset_index(drop=True)], axis=1)
        test_data_combined = pd.concat([pd.DataFrame(X_test_encoded_df), test_data[wavelength_cols].reset_index(drop=True)], axis=1)
    else:
        train_data_combined = train_data
        test_data_combined = test_data  
        
    """ if apply_remove_outliers:
        
        non_outlier_indices = remove_outliers_mahalanobis(train_data_combined[wavelength_cols]).index
        train_data_combined = train_data_combined.loc[non_outlier_indices].reset_index(drop=True)
        print(f"After Mahalanobis outlier removal, train data shape: {train_data_combined.shape}")  """
        
    if apply_remove_outliers:
        # Applying Gaussian Mixture Model (GMM) for outlier detection
        def remove_outliers_with_gmm(data, cols):
            gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
            gmm.fit(data[cols])
            probabilities = gmm.predict_proba(data[cols])
            
            # Mark samples with low probabilities as outliers
            outlier_flag = probabilities.max(axis=1) < 0.5  # Threshold for outliers
            return data[~outlier_flag]  # Keep only inliers

        # Applying Bayesian outlier detection
        def remove_outliers_with_bayesian(data, cols):
            z_scores = (data[cols] - data[cols].mean()) / data[cols].std()
            probabilities = norm.cdf(z_scores)  # Assuming a Gaussian distribution

            # Mark samples with low likelihood as outliers
            outlier_flag = (probabilities < 0.01).any(axis=1) | (probabilities > 0.99).any(axis=1)
            return data[~outlier_flag]  # Keep only inliers

        # Remove outliers using GMM or Bayesian
        train_data_combined = remove_outliers_with_gmm(train_data_combined, wavelength_cols)
        #train_data_combined = remove_outliers_with_bayesian(train_data_combined, wavelength_cols) 
        
    if apply_savgol: 
        # Apply Savitzky-Golay filter
        spectrum_train = train_data_combined[wavelength_cols]
        spectrum_test = test_data_combined[wavelength_cols]
        
        spectrum_train_filtered = pd.DataFrame(savgol_filter(spectrum_train, 7, 3, deriv=2, axis=1), columns=wavelength_cols)
        spectrum_test_filtered = pd.DataFrame(savgol_filter(spectrum_test, 7, 3, deriv=2, axis=1), columns=wavelength_cols)
        
        # Standardize the filtered spectrum
        spectrum_train_filtered_standardized = pd.DataFrame(zscore(spectrum_train_filtered, axis=1), columns=wavelength_cols)
        spectrum_test_filtered_standardized = pd.DataFrame(zscore(spectrum_test_filtered, axis=1), columns=wavelength_cols)
        
        train_data_combined[wavelength_cols] = spectrum_train_filtered_standardized
        test_data_combined[wavelength_cols] = spectrum_test_filtered_standardized    
          
    if apply_scaling:
         # Standardisers
        train_data_std = StandardScaler().fit(train_data_combined[wavelength_cols].values)

        # Standardise the data
        wavelength_train_scaled, wavelength_test_scaled = map(
            lambda data, std_mach: std_mach.transform(data),
            [
                train_data_combined[wavelength_cols].values,
                test_data_combined[wavelength_cols].values,
            ],
            [train_data_std, train_data_std],
        )     
        
        train_data_combined[wavelength_cols] = pd.DataFrame(wavelength_train_scaled, columns=wavelength_cols)
        test_data_combined[wavelength_cols] = pd.DataFrame(wavelength_test_scaled, columns=wavelength_cols)  
        
        
    if apply_robust_scaling:
        # Robust Scaler
        robust_scaler = RobustScaler()
        train_data_combined[wavelength_cols] = robust_scaler.fit_transform(train_data_combined[wavelength_cols])
        test_data_combined[wavelength_cols] = robust_scaler.transform(test_data_combined[wavelength_cols])
        
    if apply_pca:
        # Perform PCA on scaled wavelength columns
        pca = PCA(n_components=5)
        wavelength_cols = train_data_combined.columns[54:]
        
        X_train_pca = pca.fit_transform(train_data_combined[wavelength_cols])
        X_test_pca = pca.transform(test_data_combined[wavelength_cols])

        # Combine PCA components with original data
        X_train_combined = pd.concat([train_data_combined.iloc[:, :54].reset_index(drop=True), 
                                      pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(5)])], axis=1)
        X_test_combined = pd.concat([test_data_combined.iloc[:, :54].reset_index(drop=True), 
                                     pd.DataFrame(X_test_pca, columns=[f'PC{i+1}' for i in range(5)])], axis=1)
        
        train_data_combined = X_train_combined
        test_data_combined = X_test_combined
        
    """ if apply_random_forest:
        # Apply Random Forest for feature selection
        wavelength_cols = train_data_combined.columns[50:] 
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        y_train = train_data['PURITY'].iloc[train_data_combined.index]
        rf.fit(train_data_combined[wavelength_cols], y_train)
        
        # Select features based on importance
        selector = SelectFromModel(rf, threshold="mean", prefit=True)
        train_data_combined = pd.DataFrame(selector.transform(train_data_combined[wavelength_cols]), 
                                           columns=train_data_combined[wavelength_cols].columns[selector.get_support()])
        test_data_combined = pd.DataFrame(selector.transform(test_data_combined[wavelength_cols]), 
                                          columns=test_data_combined[wavelength_cols].columns[selector.get_support()])
        print(f"Shape after Random Forest feature selection: {train_data_combined.shape}")  """
    
    if apply_variance_threshold:
        # Apply VarianceThreshold
        selector = VarianceThreshold(threshold=0.05)
        train_data_combined = pd.DataFrame(selector.fit_transform(train_data_combined), columns=train_data_combined.columns[selector.get_support(indices=True)])
        test_data_combined = pd.DataFrame(selector.transform(test_data_combined), columns=test_data_combined.columns[selector.get_support(indices=True)])
        print(f"Shape after VarianceThreshold: {train_data_combined.shape}")
    
    #Aussi tester Random forest à la placde de correlation matrix
    if apply_correlation:
    # Compute correlation matrix only for wavelength columns
        #wavelength_cols = train_data_combined.columns[50:] 
        #correlation_matrix = train_data_combined[wavelength_cols].corr()
        correlation_matrix = train_data_combined.corr()

        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
        plt.title("Correlation Matrix for All Features")
        plt.savefig('/Users/maelysclerget/Desktop/ML/bio322_project/plots/correlation_matrix.jpg')  
        plt.show()

        # Identify highly correlated features (e.g., |r| > 0.999)
        threshold_high = 0.9999

        print(f"Number of features before removing highly correlated features: {train_data_combined.shape[1]}")
        
        high_corr_pairs = [
            (i, j)
            for i in range(correlation_matrix.shape[0])
            for j in range(i + 1, correlation_matrix.shape[1])
            if abs(correlation_matrix.iloc[i, j]) > threshold_high
        ]
        
        features_to_drop = set()
        for i, j in high_corr_pairs:
            features_to_drop.add(correlation_matrix.columns[j])  # Arbitrarily drop the second feature in the pair

        # Remove the selected features
        train_data_combined = train_data_combined.drop(columns=list(features_to_drop))
        test_data_combined = test_data_combined.drop(columns=list(features_to_drop))
        
        #wavelength_cols = train_data_combined.columns[50:]
        
        print(f"Number of features after removing highly correlated features: {train_data_combined.shape[1]}")
        print("Highly correlated features:")
        for i, j in high_corr_pairs:
            print(f"{correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]}")
            
        
    if apply_random_forest:
        
        wavelength_cols = train_data_combined.columns[50:]
        X_train_rf = train_data_combined[wavelength_cols]
        y_train_rf = train_data['PURITY'].iloc[train_data_combined.index].squeeze()

        # Create and train random forest model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_rf, y_train_rf)

        # Perform feature selection using the specified threshold
        sfm = SelectFromModel(rf_model, threshold=0.0048, prefit=True)
        selected_features = sfm.get_support()
        selected_feature_names = X_train_rf.columns[selected_features]

        # Apply feature selection to train and test data
        train_data_selected = train_data_combined[selected_feature_names]
        test_data_selected = test_data_combined[selected_feature_names]

        # Add back non-wavelength columns if needed
        non_wavelength_cols = [col for col in train_data_combined.columns if col not in wavelength_cols]
        train_data_combined = pd.concat([train_data_combined[non_wavelength_cols], train_data_selected], axis=1)
        test_data_combined = pd.concat([test_data_combined[non_wavelength_cols], test_data_selected], axis=1)

        print(f"Selected {len(selected_feature_names)} features using Random Forest with threshold {0.0048}.")

        # Evaluate feature importances
        feature_importances = rf_model.feature_importances_

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importances)), feature_importances)
        plt.title("Feature Importances")
        plt.xlabel("Feature Index")
        plt.ylabel("Importance Score")
        plt.show()

        """ # Test different thresholds
        thresholds = [0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.0048, 0.005, 0.0052]
        cross_val_scores = []

        for threshold in thresholds:
            # Select features based on threshold
            sfm = SelectFromModel(rf_model, threshold=threshold, prefit=True)
            selected_features = sfm.get_support()
            selected_feature_names = X_train_rf.columns[selected_features]

            # Subset the dataset
            X_train_selected = X_train_rf[selected_feature_names]

            # Compute cross-validation scores
            scores = cross_val_score(rf_model, X_train_selected, y_train_rf, cv=5, scoring='r2')
            mean_score = scores.mean()
            cross_val_scores.append(mean_score)

            print(f"Threshold: {threshold}")
            print(f"Number of selected features: {len(selected_feature_names)}")
            print(f"Cross-validated R^2 score: {mean_score:.4f}")

        # Plot cross-validated R^2 scores vs. thresholds
        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, cross_val_scores, marker='o')
        plt.title("Cross-Validated R^2 Score vs. Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Mean R^2 Score")
        plt.grid()
        plt.show() """


        """ # we obtain the names of the unwanted features
        dropped_feature_names = X_train_rf.columns[feature_indices]
        
        train_data_combined = train_data_combined.drop(columns=dropped_feature_names)
        test_data_combined = test_data_combined.drop(columns=dropped_feature_names) """

            
    # Add sample_name column back to the combined DataFrames
    train_data_combined.insert(0, 'sample_name', train_data_og['sample_name'])
    test_data_combined.insert(0, 'sample_name', test_data_og['sample_name'])
    y_train = train_data['PURITY'].iloc[train_data_combined.index]

    print(f"Shape of OG train data: {train_data_og.shape}")
    print(f"Shape of OG test data: {test_data_og.shape}")
    print(f"Shape of train data: {train_data_combined.shape}")
    print(f"Shape of test data: {test_data_combined.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    
    train_data_combined.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/train_data_combined.csv', index=False)
    print('Submission file saved successfully.')
    test_data_combined.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/test_data_combined.csv', index=False)
    print('Submission file saved successfully.')
            
    return train_data_combined, test_data_combined, y_train

def apply_log_transformation(y):
    return np.log(y)

def submission_file(y_test_predicted):
    submission_reduced = pd.DataFrame({
        'ID': range(1, len(y_test_predicted) + 1),
        'PURITY': y_test_predicted
    })
    return submission_reduced

def calculate_feature_importance(X_train, y_train, X_test, threshold=0.25):
    # Calculate feature importance using a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print feature importance
    feature_importance = pd.Series(model.coef_, index=X_train.columns)
    feature_importance = feature_importance.abs().sort_values(ascending=False)
    wavelength_feature_importance_df = feature_importance.reset_index()
    wavelength_feature_importance_df.columns = ['Feature', 'Importance']
    wavelength_feature_importance_df.to_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/feature_importance_LR1.csv', index=False)
    print('Feature Importance saved successfully.')
    
    # Calculate stats threshold
    threshold_value = feature_importance.quantile(threshold)
    
    # Identify low-importance features
    low_importance_features = feature_importance[feature_importance < threshold_value].index
    print(f'Low importance features: {low_importance_features}')
    
    # Remove low-importance features
    X_train_reduced = X_train.drop(columns=low_importance_features)
    X_test_reduced = X_test.drop(columns=low_importance_features)
    
    return X_train_reduced, X_test_reduced

def main():
    preprocessing_v1(apply_one_hot=True, apply_scaling=True, apply_pca=False, apply_correlation=True, apply_remove_outliers=False, apply_variance_threshold=False, apply_random_forest=False, apply_robust_scaling=False)
    
if __name__ == '__main__':
    main()