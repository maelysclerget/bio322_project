import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import BayesianRidge
from bayes_opt import BayesianOptimization
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm 
from scipy.stats import chi2
from preprocessing import preprocessing_v1, apply_log_transformation

def visualisation():
    
   substances_og = pd.read_csv('/Users/maelysclerget/Desktop/ML/bio322_project/epfl-bio-322-2024/substances.csv')
    
    # Filter substances with more than 10 samples
   substance_counts = substances_og['substance'].value_counts()
   filtered_substances = substance_counts[substance_counts > 10].index

    # Plot the filtered substances
   filtered_substances_counts = substance_counts[filtered_substances]
   filtered_substances_counts.plot(kind='bar')
   plt.title('Number of samples per substance (more than 10 samples)')
   plt.xlabel('Substance')
   plt.ylabel('Number of samples')
   plt.show()
   
   for substance in filtered_substances:
        subset = substances_og[substances_og['substance'] == substance]
        plt.plot(subset.iloc[:, 1:].mean(), label=substance)
   plt.legend()
   plt.title('Average Spectral Patterns of Substances')
   plt.show()

def main():
    visualisation()
    
if __name__ == '__main__':
    main()