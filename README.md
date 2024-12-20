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
measure_type_display and prod_substance 
sample_name helped us understand and compare the wavelengths measurements even if it's the same sample only with a different 

- Analyse of purity distribution : 


- Analyse of wavelengths : 
(exécute dans le fichier data_vis) 
This plot is a plot of the wavelengths results in function of the wavelengths for 5 different wavelengths. This is helpfull to determine if we can easily detect outliers. Here we can see that for most of the different wavelengths, the results are in the area [-0.4, 0.6]. All the wavelengths out of this area are outliers. 

- Analyse of the same samples, different substances or devices and their graphs : 
(exécute dans le fichier data_Vis2) 
These plots were necessary to check the importance of 2 categorical columns : substance_form_display and device_serial. With those plots, we could clearly see 2 or more curves for each sample that have been measured a few times in different substances of devices. In most of the cases, the curves are different for the different devices. We got to the conclusion that the substances and devices were important in the results of the wavelengths and then in the results of the purity.  

During the first part of the data visualisation, we also got into the analysis of the substances dataset. It picks up other drugs like cocaine, MDMA or caffeine and their chromatography results at different wavelengths. The size of the substances data set is (1432, 126) and we got to plot the different categories and their frequence. 


- **`preprocessing.ipynb`**: 
Clean and preprocess the data, including handling missing values, feature engineering, and encoding categorical variables.


+ preprocessing of susbtances : 
(fichier substances_new_data -> merge.csv)
We wanted to analyse and train our models with the additionnal data set. To do this, we tried to concatenate the 2 (train and substances) datasets. 
In the first place, we gave the same column names to the 2 data sets. For this, we rounded the wavelengths column names for the substances data set with a precision=1. Then we added a purity column to the substances data set with only 0s. In fact a cocaine sample cannot have a purity in heroine different than 0.
To fill out the other categorical columns, we calculated the distribution of each categories in the train. For example, the distribution of the substance_form_display() in 'Homogenized Powder', 'Non Homogenized Powder' or'Unspecified'. Then, we created and filled out the substances columns respecting the distributions of probabilities. 
We finally concaneted the 2 data sets to have a main data set with the same number of columns and 2865 lines.


- **`train_linear.ipynb`**: 
Train and evaluate a baseline linear model, experimenting with regularization techniques. Several models have been chosen to perform training methods and final results on the data. 

- **`train_non_linear.ipynb`**: 
Train and tune advanced non-linear models like Random Forests and XGBoost to discuss the predictions accuracy of the dataset.









