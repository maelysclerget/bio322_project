
import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
import preprocessing 

param_grid = {
    'colsample_bytree': np.linspace(0.5, 1, 5),
    'subsample': np.linspace(0.5, 1, 5),
    'max_depth': np.arange(2, 7, 1)
}


X_train, X_test, y_train = preprocessing(apply_one_hot=True, apply_scaling=True)

model = XGBRegressor(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
