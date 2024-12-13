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
from sklearn.model_selection import cross_val_score
    
def scoring(y_pred, y_true):
    err = np.abs(y_pred - y_true)
    frac = (err <= 5).astype(int)
    return np.mean(frac)

def plot_raw_vs_filtered(raw_signal, filtered_signal, wavelength_cols, sample_idx=0):
    """
    Plots the raw and filtered signal for comparison.
    
    Args:
        raw_signal: DataFrame of the raw signal.
        filtered_signal: DataFrame of the filtered signal.
        wavelength_cols: List of wavelength columns.
        sample_idx: Index of the sample to plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(wavelength_cols, raw_signal.iloc[sample_idx, :], label="Raw Signal", alpha=0.7)
    plt.plot(wavelength_cols, filtered_signal.iloc[sample_idx, :], label="Filtered Signal", linestyle='--')
    plt.title(f"Sample {sample_idx}: Raw vs. Filtered Signal")
    plt.xlabel("Wavelength")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid()
    plt.savefig('savitzky_golay_filter.png')
    plt.show()

def preprocessing_v1(apply_one_hot=False, apply_scaling=False, apply_pca=False, 
                     apply_correlation=False, apply_remove_outliers=False, 
                     apply_variance_threshold=False, apply_random_forest=False, 
                     apply_robust_scaling=False, apply_savgol=False):   
    train_data_og = pd.read_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/train.csv')
    test_data_og = pd.read_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/test.csv')
   
    train_data = train_data_og.copy()
    test_data = test_data_og.copy()
    
    train_data = train_data.drop(columns=['prod_substance','measure_type_display'])
    test_data = test_data.drop(columns=['prod_substance','measure_type_display'])
    
    non_wavelength_cols = ['device_serial', 'substance_form_display']
    wavelength_cols = train_data.columns[4:]
    
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
        
        # Multiply by 3 for "Non homogenized powder" and by 1.4 for "Unspecified"
        if 'substance_form_display_Non homogenized powder' in X_train_encoded_df.columns:
            X_train_encoded_df['substance_form_display_Non homogenized powder'] *= 3
            X_test_encoded_df['substance_form_display_Non homogenized powder'] *= 3
        
        if 'substance_form_display_Unspecified' in X_train_encoded_df.columns:
            X_train_encoded_df['substance_form_display_Unspecified'] *= 1.4
            X_test_encoded_df['substance_form_display_Unspecified'] *= 1.4
            
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

        def remove_outliers(dataframe, columns_to_check, threshold=1.5, outlier_percentage=0.10):
            """
            Removes rows from the dataset if more than a certain percentage of values in specified columns are outliers.

            Parameters:
                dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.
                columns_to_check (list): List of columns to check for outliers.
                threshold (float): The IQR multiplier to define outliers (default is 1.5).
                outlier_percentage (float): The percentage of outlier values in the specified columns to consider for removal (default is 0.5).

            Returns:
                pd.DataFrame: The cleaned DataFrame with rows containing too many outliers removed.
            """
            # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the specified columns
            Q1 = dataframe[columns_to_check].quantile(0.25)
            Q3 = dataframe[columns_to_check].quantile(0.75)
            IQR = Q3 - Q1
            initial_rows = dataframe.shape[0]
            
            # Determine outliers for the specified columns
            outliers = ((dataframe[columns_to_check] < (Q1 - threshold * IQR)) | (dataframe[columns_to_check] > (Q3 + threshold * IQR)))
            # Calculate the percentage of outliers in each row for the specified columns
            outlier_counts = outliers.sum(axis=1) / len(columns_to_check)
            # Keep rows where the percentage of outliers is less than or equal to the threshold
            cleaned_dataframe = dataframe[outlier_counts <= outlier_percentage]
            removed_rows = initial_rows - cleaned_dataframe.shape[0]
            print(f"Removed {removed_rows} rows where more than {outlier_percentage * 100}% of columns are outliers.")

            return cleaned_dataframe

        # Remove outliers from the numerical columns in both train and test data
        wavelength_cols = train_data_combined.columns[49:]
        train_data_combined = remove_outliers(train_data_combined, wavelength_cols)
        test_data_combined = remove_outliers(test_data_combined, wavelength_cols, outlier_percentage=0.15)  
        
    if apply_savgol: 
        # Apply Savitzky-Golay filter
        spectrum_train = train_data_combined[wavelength_cols]
        spectrum_test = test_data_combined[wavelength_cols]
        
        spectrum_train_filtered = pd.DataFrame(savgol_filter(spectrum_train, 7, 3, deriv=2, axis=1), columns=wavelength_cols)
        spectrum_test_filtered = pd.DataFrame(savgol_filter(spectrum_test, 7, 3, deriv=2, axis=1), columns=wavelength_cols)
        
        # Standardize the filtered spectrum
        spectrum_train_filtered_standardized = pd.DataFrame(zscore(spectrum_train_filtered, axis=1), columns=wavelength_cols)
        spectrum_test_filtered_standardized = pd.DataFrame(zscore(spectrum_test_filtered, axis=1), columns=wavelength_cols)
        
        plot_raw_vs_filtered(spectrum_train, spectrum_train_filtered, wavelength_cols, sample_idx=0)

        train_data_combined[wavelength_cols] = spectrum_train_filtered_standardized
        test_data_combined[wavelength_cols] = spectrum_test_filtered_standardized 
         
    if apply_remove_outliers:

        def remove_outliers(dataframe, columns_to_check, threshold=1.8, outlier_percentage=0.70):
            """
            Removes rows from the dataset if more than a certain percentage of values in specified columns are outliers.

            Parameters:
                dataframe (pd.DataFrame): The input dataset as a pandas DataFrame.
                columns_to_check (list): List of columns to check for outliers.
                threshold (float): The IQR multiplier to define outliers (default is 1.5).
                outlier_percentage (float): The percentage of outlier values in the specified columns to consider for removal (default is 0.10).

            Returns:
                pd.DataFrame: The cleaned DataFrame with rows containing too many outliers removed.
            """
            # Calculate Q1, Q3, and IQR for the specified columns
            Q1 = dataframe[columns_to_check].quantile(0.25)
            Q3 = dataframe[columns_to_check].quantile(0.75)
            IQR = Q3 - Q1

            # Define outlier thresholds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Identify outliers for each column
            is_outlier = pd.DataFrame(False, index=dataframe.index, columns=columns_to_check)
            for column in columns_to_check:
                is_outlier[column] = (dataframe[column] < lower_bound[column]) | (dataframe[column] > upper_bound[column])

            # Calculate the proportion of outlier columns for each row
            outlier_proportion = is_outlier.sum(axis=1) / len(columns_to_check)

            # Keep rows where the proportion of outliers is below the specified threshold
            cleaned_dataframe = dataframe[outlier_proportion <= outlier_percentage]

            # Print number of rows removed for clarity
            removed_rows = dataframe.shape[0] - cleaned_dataframe.shape[0]
            print(f"Removed {removed_rows} rows with more than {outlier_percentage * 100}% outliers in specified columns.")
            
            return cleaned_dataframe

        # Example Usage
        wavelength_cols = train_data_combined.columns[49:]  # Example: Adjust column index as per dataset
        train_data_combined = remove_outliers(train_data_combined, wavelength_cols)
        
    
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
        plt.savefig('feature_importances.png')
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
        plt.title("Optimal Feature Selection Threshold vs. Model Performance")
        plt.xlabel("Feature Importance Threshold")
        plt.ylabel("Mean Cross-Validated R² Score")
        plt.grid()
        plt.savefig('threshold_RF.png')
        plt.show() """


        """ # we obtain the names of the unwanted features
        dropped_feature_names = X_train_rf.columns[feature_indices]
        
        train_data_combined = train_data_combined.drop(columns=dropped_feature_names)
        test_data_combined = test_data_combined.drop(columns=dropped_feature_names) """

    
    # Add sample_name column back to the combined DataFrames
    train_data_combined = pd.concat([pd.DataFrame({'sample_name': train_data_og['sample_name']}), train_data_combined], axis=1)
    test_data_combined = pd.concat([pd.DataFrame({'sample_name': test_data_og['sample_name']}), test_data_combined], axis=1)
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
    preprocessing_v1(apply_one_hot=True, apply_scaling=False, apply_pca=False, apply_correlation=False, apply_remove_outliers=False, apply_variance_threshold=False, apply_random_forest=False, apply_savgol=True)
if __name__ == '__main__':
    main()