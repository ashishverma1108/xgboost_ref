#!/usr/bin/env python3

"""
 Project : xgboost reference
 Module : setup for model training
 Description : This code sets up libraries, imports dataset, and breaks it
 into traing and test
"""

## Module imports
import pandas as pd
from sklearn.datasets import load_boston
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np

## Loading data... for this example using boston housing data
boston = load_boston()

## Converting to pandas dataframe
data_df = pd.DataFrame(boston.data)
## Add the target variable to the dataframe. This is provided separately in
## default boston dataset
data_df["price"] = boston.target

## Above two steps are helpful because they help provide and retain col names
X, y = data_df.iloc[:,:-1], data_df.iloc[:,-1]

## Setting final matrix to be used for downstream processing
data_matrix = xgb.DMatrix(data = X, label = y)

## Split datasets in to train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
