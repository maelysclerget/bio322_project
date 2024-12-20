# Bio322_project
ML project for course bio322

BIO-322 Machine Learning project 

## Contributors

- [Alice Priolet](https://github.com/alicepriolet)
- [Maëlys Clerget](https://github.com/maelysclerget)

Introduction : 
This repository contains the code and documentation for the miniproject. The task focuses on predicting the purity level of heroin samples using machine learning techniques applied to infrared spectrum data. This prediction can provide a faster and cost-effective alternative to traditional chromatography method for the detection of heroin in samples. This project explores cost-effective and real-time solutions using machine learning models to analyze data from portable infrared spectrometry devices. The competition of the different projects accuracy is hosted on Kaggle.

The project is organized into several sections to ensure a systematic approach, reproducibility, and ease of understanding.


## Table of contents 

- **`data_visualization.ipynb`**: 
Load, explore, and visualize the provided dataset to gain insights and identify patterns in the data. We will be able to analyse the shape of the dataset or the relevant informations. 

- Analyse of the categorical columns (different plots) :


- **`preprocessing.ipynb`**: 
Clean and preprocess the data, including handling missing values, feature engineering, and encoding categorical variables.

- **`train_linear.ipynb`**: 
Train and evaluate a baseline linear model, experimenting with regularization techniques. Several models have been chosen to perform training methods and final results on the data. 

- **`train_non_linear.ipynb`**: 
Train and tune advanced non-linear models like Random Forests and XGBoost to discuss the predictions accuracy of the dataset.




Best accuracy: -31.707467678348735
Best parameters: {'polynomial__degree': 1}
Training MSE: 19.393135999284617
CV MSE: 31.454214389906372

Ridge avec tree 

Best accuracy: -24.92438459892387
Best parameters: {'polynomial__degree': 2, 'regression__alpha': 1e-05}
Training MSE: 13.854619801051394
CV MSE: 24.92438459892387







