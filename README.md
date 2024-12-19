# bio322_project
ML project for course bio322

BIO-322 Machine Learning project 

Team : Alice Priolet and Maëlys Clerget 

Introduction : 
This repository contains the code and documentation for the miniproject. The task focuses on predicting the purity level of heroin samples using machine learning techniques applied to infrared spectrum data. This prediction can provide a faster and cost-effective alternative to traditional chromatography method for the detection of heroin in samples.

The project is organized into several sections to ensure a systematic approach, reproducibility, and ease of understanding.

Table of contents : 

1. 	Data Inspection
Load, explore, and visualize the provided dataset to gain insights and identify patterns in the data. We will be able to analyse the shape of the dataset or the relevant informations. 

2. Preprocessing 
Clean and preprocess the data, including handling missing values, feature engineering, and encoding categorical variables.

3. Linear Models
Train and evaluate a baseline linear model, experimenting with regularization techniques. Several models have been chosen to perform training methods and final results on the data. 

4. 	Non-Linear Models
Train and tune advanced non-linear models like Random Forests and XGBoost to discuss the predictions accuracy of the dataset.

5. Summary and Conclusion


Installation and Running : 
1. Install the environment
Download and install the Anaconda environment on your computer. This will help you manage dependencies and ensure a consistent environment for the project.

Ensure the following libraries are installed in your Python environment. Use the commands below to install them: 
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

2. Clone the repository 
Clone the GitHub repository to a local folder on your computer using the following command in your terminal:
git clone <https://github.com/maelysclerget/bio322_project.git>
cd <repository_folder> 


3. Visualizing the data : 
    a. Train data : 

    b. Substances data : 

4. Prepare the data : 

5. Running the project : 


##Contributors : 
- [Alice Priolet](https://github.com/alicepriolet)
- [Maëlys Clerget](https://github.com/maelysclerget)

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

# Miniproject BIO-322: Predicting Heroin Purity
This repository contains the code, analysis, and results for the BIO-322 miniproject. The goal of this project is to predict the purity level of heroin samples using infrared spectra and machine learning techniques.

# Overview 
Confiscated drug samples often require purity analysis for legal and law enforcement purposes. Traditional methods, such as chromatography, are expensive and time-consuming. This project explores cost-effective and real-time solutions using machine learning models to analyze data from portable infrared spectrometry devices. The competition is hosted on Kaggle.

## Authors
Maëlys Clerget & Alice Priolet 

## Code organisation 

- **`data_visualization.ipynb`**: 
- **`preprocessing.ipynb`**: 
- **`train_linear.ipynb`**: 
- **`train_non_linear.ipynb`**: 

