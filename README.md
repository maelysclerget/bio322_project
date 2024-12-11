# bio322_project
ML project for course bio322

Best accuracy: -31.707467678348735
Best parameters: {'polynomial__degree': 1}
Training MSE: 19.393135999284617
CV MSE: 31.454214389906372
Submission file saved successfully.

Ridge avec tree 

Best accuracy: -24.92438459892387
Best parameters: {'polynomial__degree': 2, 'regression__alpha': 1e-05}
Training MSE: 13.854619801051394
CV MSE: 24.92438459892387

OMP: 
Best accuracy: -6.2257053452506685
Best parameters: {'polynomial__degree': 2, 'regression__n_nonzero_coefs': 51}
Training MSE: 3.8104301463852357
CV MSE: 6.2257053452506685

# Miniproject BIO-322: Predicting Heroin Purity
This repository contains the code, analysis, and results for the BIO-322 miniproject. The goal of this project is to predict the purity level of heroin samples using infrared spectra and machine learning techniques.

# Overview 
Confiscated drug samples often require purity analysis for legal and law enforcement purposes. Traditional methods, such as chromatography, are expensive and time-consuming. This project explores cost-effective and real-time solutions using machine learning models to analyze data from portable infrared spectrometry devices. The competition is hosted on Kaggle.

# Authors
Maëlys Clerget & Alice Priolet 

# Code organisation 

- **`data_visualization.py`**: 
- **`preprocessing.py`**: 
- **`bayesian_ridge_regression`**: 
- **`elasticnet_regression`**: 

# Data Visualization 

## Correlation matrix 
![Correlation matrix]('/Users/maelysclerget/Desktop/ML/bio322_project/plots/correlation_matrix.jpg')
A coolwarm colormap highlights strong correlations (red for positive, blue for negative) between features. Features with correlations above 0.9999 were removed, reducing the feature count from 175 to 126. For example, 1001 and 1007.2 had a correlation of 0.9999021664382056

## Response variable purity 
![Purity]('/Users/maelysclerget/Desktop/ML/bio322_project/plots/response_variable.png')
![Purity with log transformation applied]('/Users/maelysclerget/Desktop/ML/bio322_project/plots/response_variable_log.png')




